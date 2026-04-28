###########################################
# Forward Forward - Simple Shapes (Fixed)
###########################################

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ─────────────────────────────────────────
# INPUT
# ─────────────────────────────────────────
print("1: Centered Shapes")
print("2: Not Centered Shapes")
print("3: Centered noisy Shapes")
print("4: Not Centered noisy Shapes")
choice = input("Select dataset (1–4): ")

if choice == '1':
    data_path = r"D:\dataset\simple_shapes_centered"
elif choice == '2':
    data_path = r"D:\dataset\simple_shapes"
elif choice == '3':
    data_path = r"D:\dataset\simple_shapes_centered_noisy"
else:
    data_path = r"D:\dataset\simple_shapes_off_centered_noisy"

epochs    = int(input("Enter number of epochs: "))
lr        = float(input("Enter learning rate: "))
threshold = float(input("Enter threshold: "))

NUM_CLASSES = 5

# ─────────────────────────────────────────
# DATA
# ─────────────────────────────────────────
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.ImageFolder(root=data_path + "\\train", transform=transform)
test_data  = datasets.ImageFolder(root=data_path + "\\test",  transform=transform)

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes: {train_data.classes}")

# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def make_input(x, y):
    x = x.view(x.size(0), -1)
    y_oh = F.one_hot(y, NUM_CLASSES).float() * 2 - 1
    x = x.clone()
    x[:, :NUM_CLASSES] = y_oh
    return x

# ✅ FIX 1: neg_labels — guaranteed different label via random offset
def neg_labels(y):
    offsets = torch.randint(1, NUM_CLASSES, y.shape)
    return (y + offsets) % NUM_CLASSES

def count_dead_neurons(h):
    return (h.mean(dim=0) <= 0.0).sum().item()

def per_class_accuracy(net, loader):
    correct_per = [0] * NUM_CLASSES
    total_per   = [0] * NUM_CLASSES
    for x, y in loader:
        preds = net.predict(x)
        for c in range(NUM_CLASSES):
            mask = (y == c)
            correct_per[c] += (preds[mask] == y[mask]).sum().item()
            total_per[c]   += mask.sum().item()
    return [(correct_per[c], total_per[c]) for c in range(NUM_CLASSES)]

def confusion_matrix(net, loader):
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        preds = net.predict(x)
        for t, p in zip(y, preds):
            matrix[t][p] += 1
    return matrix

def print_confusion(matrix, class_names):
    print("\n  Confusion matrix:")
    print(f"  {'':12}" + "".join(f"{n:>10}" for n in class_names))
    for i, row in enumerate(matrix):
        print(f"  {class_names[i]:<12}" + "".join(f"{v.item():>10}" for v in row))

def print_diagnostic_block(epoch, stored, class_names):
    print(f"\n[Epoch {epoch}]")
    print(f"  Loss: {stored['loss']:.2f} | Test Acc: {stored['acc']:.2f}%")

    ll  = stored['layer_loss']
    gp  = stored['avg_gp']
    gn  = stored['avg_gn']
    g   = stored['gap']
    gn2 = stored['avg_gnorm']
    hp  = stored['hit_p_pct']
    hn  = stored['hit_n_pct']
    d   = stored['avg_dead']

    print(f"  Layer losses  →  L1: {ll[0]:>7.2f}   L2: {ll[1]:>7.2f}   L3: {ll[2]:>7.2f}")
    print(f"  Goodness g+   →  L1: {gp[0]:>6.3f}    L2: {gp[1]:>6.3f}    L3: {gp[2]:>6.3f}")
    print(f"  Goodness g-   →  L1: {gn[0]:>6.3f}    L2: {gn[1]:>6.3f}    L3: {gn[2]:>6.3f}")
    print(f"  Goodness gap  →  L1: {g[0]:>6.3f}    L2: {g[1]:>6.3f}    L3: {g[2]:>6.3f}   {stored['gap_note']}")
    print(f"  Grad norm     →  L1: {gn2[0]:>6.3f}    L2: {gn2[1]:>6.3f}    L3: {gn2[2]:>6.3f}")
    print(f"  Threshold hit →  g+: {hp[0]:>5.1f}%   g-: {hn[0]:>5.1f}%   (L1 shown)")
    print(f"  Dead neurons  →  L1: {d[0]:>4d}     L2: {d[1]:>4d}     L3: {d[2]:>4d}")

    print("\n  Per-class accuracy:")
    for i, name in enumerate(class_names):
        c, t = stored['per_class'][i]
        print(f"    {name:<12}: {100*c/max(t,1):.2f}% ({c}/{t})")

    print_confusion(stored['cm'], class_names)

# ─────────────────────────────────────────
# VISUALISATION  (thesis-quality graphs)
# ─────────────────────────────────────────
def plot_all(epoch_store, class_names, epochs, best_epoch):
    ep = list(range(1, epochs + 1))

    losses      = [epoch_store[e]['loss']      for e in ep]
    train_accs  = [epoch_store[e]['train_acc'] for e in ep]
    test_accs   = [epoch_store[e]['acc']       for e in ep]
    gaps        = [[epoch_store[e]['gap'][l]   for e in ep] for l in range(3)]
    dead        = [[epoch_store[e]['avg_dead'][l] for e in ep] for l in range(3)]
    final_pc    = [100 * epoch_store[epochs]['per_class'][i][0] /
                   max(epoch_store[epochs]['per_class'][i][1], 1)
                   for i in range(NUM_CLASSES)]
    final_cm    = epoch_store[epochs]['cm'].numpy()

    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']
    layer_colors = ['#E63946', '#2A9D8F', '#E9C46A']

    plt.rcParams.update({
        'font.family':     'DejaVu Sans',
        'font.size':       11,
        'axes.titlesize':  13,
        'axes.titleweight':'bold',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.dpi':      150,
    })

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Forward-Forward Network — Simple Shapes\n"
        f"Epochs: {epochs}  |  LR: {lr}  |  Threshold: {threshold}  |  "
        f"Train: {len(train_data)}  |  Test: {len(test_data)}",
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # ── 1. Loss curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, losses, color='#2E86AB', linewidth=2, marker='o', markersize=5)
    ax1.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7,
                label=f'Best epoch {best_epoch}')
    ax1.set_title("Total Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)
    ax1.set_xticks(ep)

    # ── 2. Train vs Test accuracy ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, train_accs, color='#2A9D8F', linewidth=2, marker='o',
             markersize=5, label='Train')
    ax2.plot(ep, test_accs,  color='#E63946', linewidth=2, marker='s',
             markersize=5, label='Test')
    ax2.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title("Train vs Test Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.legend(fontsize=9)
    ax2.set_xticks(ep)

    # ── 3. Goodness gap per layer ─────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    for l in range(3):
        ax3.plot(ep, gaps[l], color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax3.axhline(0.3, color='gray', linestyle=':', linewidth=1, alpha=0.6,
                label='Gap=0.3 (small)')
    ax3.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.6,
                label='Gap=0.8 (clear)')
    ax3.set_title("Goodness Gap per Layer\n(g⁺ − g⁻)")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Gap")
    ax3.legend(fontsize=8)
    ax3.set_xticks(ep)

    # ── 4. Dead neurons per layer ─────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    for l in range(3):
        ax4.plot(ep, dead[l], color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax4.set_title("Dead Neurons per Layer")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("Dead Neuron Count")
    ax4.legend(fontsize=9)
    ax4.set_xticks(ep)

    # ── 5. Per-class accuracy bar chart ──────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    bars = ax5.bar(class_names, final_pc, color=colors, edgecolor='white',
                   linewidth=0.8)
    for bar, val in zip(bars, final_pc):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f"{val:.1f}%", ha='center', va='bottom', fontsize=9,
                 fontweight='bold')
    ax5.set_title(f"Per-class Accuracy\n(Final Epoch {epochs})")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_ylim(0, 115)
    ax5.tick_params(axis='x', rotation=15)

    # ── 6. Confusion matrix heatmap ───────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(final_cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    ax6.set_xticks(range(NUM_CLASSES))
    ax6.set_yticks(range(NUM_CLASSES))
    ax6.set_xticklabels(class_names, rotation=30, ha='right', fontsize=9)
    ax6.set_yticklabels(class_names, fontsize=9)
    ax6.set_title(f"Confusion Matrix\n(Final Epoch {epochs})")
    ax6.set_xlabel("Predicted")
    ax6.set_ylabel("Actual")
    thresh = final_cm.max() / 2.0
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            ax6.text(j, i, str(final_cm[i, j]),
                     ha='center', va='center', fontsize=9,
                     color='white' if final_cm[i, j] > thresh else 'black')

    # ── 7. Layer-wise loss across epochs ─────────────────
    ax7 = fig.add_subplot(gs[2, 0])
    for l in range(3):
        ll_ep = [epoch_store[e]['layer_loss'][l] for e in ep]
        ax7.plot(ep, ll_ep, color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax7.set_title("Layer-wise Loss per Epoch")
    ax7.set_xlabel("Epoch")
    ax7.set_ylabel("Loss")
    ax7.legend(fontsize=9)
    ax7.set_xticks(ep)

    # ── 8. Goodness g+ and g- for each layer (final epoch) ─
    ax8 = fig.add_subplot(gs[2, 1])
    x_pos   = np.arange(3)
    gp_vals = epoch_store[epochs]['avg_gp']
    gn_vals = epoch_store[epochs]['avg_gn']
    width   = 0.35
    ax8.bar(x_pos - width/2, gp_vals, width, label='g⁺ (positive)',
            color='#2A9D8F', edgecolor='white')
    ax8.bar(x_pos + width/2, gn_vals, width, label='g⁻ (negative)',
            color='#E63946', edgecolor='white')
    ax8.axhline(threshold, color='black', linestyle='--', linewidth=1,
                label=f'Threshold={threshold}')
    ax8.set_title(f"Goodness g⁺ vs g⁻ by Layer\n(Final Epoch {epochs})")
    ax8.set_xlabel("Layer")
    ax8.set_ylabel("Goodness")
    ax8.set_xticks(x_pos)
    ax8.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
    ax8.legend(fontsize=9)

    # ── 9. Grad norm across epochs per layer ─────────────
    ax9 = fig.add_subplot(gs[2, 2])
    for l in range(3):
        gn_ep = [epoch_store[e]['avg_gnorm'][l] for e in ep]
        ax9.plot(ep, gn_ep, color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax9.set_title("Gradient Norm per Layer")
    ax9.set_xlabel("Epoch")
    ax9.set_ylabel("Grad Norm")
    ax9.legend(fontsize=9)
    ax9.set_xticks(ep)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # ── DISPLAY ───────────────────────────────────────────
    plt.show()

    # ── SAVE (commented out — uncomment to save to disk) ──
    # save_path = r"D:\results\ff_thesis_plots.png"
    # fig.savefig(save_path, dpi=300, bbox_inches='tight')
    # print(f"  Plots saved → {save_path}")


# ─────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return torch.relu(self.linear(x))

    def train_step(self, xp, xn):
        hp = self.forward(xp)
        hn = self.forward(xn)

        gp = (hp ** 2).mean(dim=1)
        gn = (hn ** 2).mean(dim=1)

        loss = (F.softplus(-gp + threshold) +
                F.softplus( gn - threshold)).mean()

        self.opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1e9)
        self.opt.step()

        # ✅ FIX 2: Normalize activations before passing to the next layer
        hp_norm = hp / (hp.norm(dim=1, keepdim=True) + 1e-8)
        hn_norm = hn / (hn.norm(dim=1, keepdim=True) + 1e-8)

        return (hp_norm.detach(), hn_norm.detach(), loss.item(),
                gp.mean().item(), gn.mean().item(), grad_norm.item(),
                hp.detach(), hn.detach())


class FFNet:
    def __init__(self):
        self.layers = [
            FFLayer(784, 500),
            FFLayer(500, 500),
            FFLayer(500, 500),
        ]

    def predict(self, x):
        x = x.view(x.size(0), -1)
        scores = []
        for c in range(NUM_CLASSES):
            y_c = torch.full((x.size(0),), c)
            xc  = make_input(x, y_c)
            h, g = xc, 0
            for layer in self.layers:
                h = layer.forward(h)
                g += (h ** 2).mean(dim=1)
                # ✅ FIX 2: Normalize between layers during inference too
                h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            scores.append(g.unsqueeze(1))
        return torch.cat(scores, dim=1).argmax(dim=1)

    def evaluate(self, loader):
        correct, total = 0, 0
        for x, y in loader:
            preds = self.predict(x)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total

    def train(self, loader, epochs):
        epoch_store = {}
        prev_loss   = None
        epoch_times = []

        for epoch in range(1, epochs + 1):
            total_loss  = 0
            t_start     = time.time()

            layer_loss  = [0.0, 0.0, 0.0]
            layer_gp    = [0.0, 0.0, 0.0]
            layer_gn    = [0.0, 0.0, 0.0]
            layer_gnorm = [0.0, 0.0, 0.0]
            layer_dead  = [0,   0,   0  ]
            layer_hit_p = [0,   0,   0  ]
            layer_hit_n = [0,   0,   0  ]
            n_batches   = 0

            for x, y in loader:
                y_neg = neg_labels(y)
                xp = make_input(x, y)
                xn = make_input(x, y_neg)

                for i, layer in enumerate(self.layers):
                    xp, xn, loss, gp_mean, gn_mean, gnorm, hp_raw, hn_raw = \
                        layer.train_step(xp, xn)

                    total_loss      += loss
                    layer_loss[i]   += loss
                    layer_gp[i]     += gp_mean
                    layer_gn[i]     += gn_mean
                    layer_gnorm[i]  += gnorm
                    layer_dead[i]   += count_dead_neurons(hp_raw)

                    gp_b = (hp_raw ** 2).mean(dim=1)
                    gn_b = (hn_raw ** 2).mean(dim=1)
                    layer_hit_p[i]  += (gp_b > threshold).sum().item()
                    layer_hit_n[i]  += (gn_b < threshold).sum().item()

                n_batches += 1

            epoch_time = time.time() - t_start
            epoch_times.append(epoch_time)

            nb       = max(n_batches, 1)
            avg_gp   = [layer_gp[i]    / nb for i in range(3)]
            avg_gn   = [layer_gn[i]    / nb for i in range(3)]
            avg_gn2  = [layer_gnorm[i] / nb for i in range(3)]
            avg_dead = [layer_dead[i]  // nb for i in range(3)]
            gap      = [avg_gp[i] - avg_gn[i] for i in range(3)]

            total_samples = len(train_data) * 1.0
            hit_p_pct = [100 * layer_hit_p[i] / (total_samples * nb + 1e-8) for i in range(3)]
            hit_n_pct = [100 * layer_hit_n[i] / (total_samples * nb + 1e-8) for i in range(3)]

            test_acc  = self.evaluate(test_loader)
            train_acc = self.evaluate(loader)

            cm        = confusion_matrix(self, test_loader)
            per_class = per_class_accuracy(self, test_loader)

            # ── gap note (before epoch_store) ──
            gap_note = ""
            if max(gap) < 0.3:
                gap_note = "← gap small"
            elif min(gap) > 0.8:
                gap_note = "← clear separation"

            epoch_store[epoch] = {
                'loss':       total_loss,
                'acc':        test_acc,
                'train_acc':  train_acc,
                'cm':         cm,
                'per_class':  per_class,
                'layer_loss': layer_loss,
                'avg_gp':     avg_gp,
                'avg_gn':     avg_gn,
                'gap':        gap,
                'gap_note':   gap_note,
                'avg_gnorm':  avg_gn2,
                'hit_p_pct':  hit_p_pct,
                'hit_n_pct':  hit_n_pct,
                'avg_dead':   avg_dead,
            }

            # ── ΔLoss & ETA ──
            delta_str = f"{total_loss - prev_loss:+.2f}" if prev_loss is not None else "—"
            prev_loss = total_loss
            avg_time  = sum(epoch_times) / len(epoch_times)
            remaining = avg_time * (epochs - epoch)
            converged_tag = " ← CONVERGED" if test_acc == 100.0 else ""

            # ── EPOCH PROGRESS LINE ──
            print(f"Epoch {epoch:>3} | Loss: {total_loss:>8.2f} | ΔLoss: {delta_str:>9} | "
                  f"Train Acc: {train_acc:>6.2f}% | Test Acc: {test_acc:>6.2f}% | "
                  f"Time: {epoch_time:>5.1f}s | ETA: {remaining:>5.1f}s{converged_tag}")

        # ───────── FULL DIAGNOSTICS ─────────
        print("\n===== FULL DIAGNOSTICS =====")
        print(f"Dataset Choice : {choice}")
        print(f"Epochs         : {epochs}")
        print(f"Learning Rate  : {lr}")
        print(f"Threshold      : {threshold}")
        print(f"Num Classes    : {NUM_CLASSES}")
        print(f"Classes        : {train_data.classes}")
        print(f"Train Samples  : {len(train_data)}")
        print(f"Test Samples   : {len(test_data)}")
        print(f"Architecture   : 784 → 500 → 500 → 500")
        print(f"Optimizer      : Adam")
        print(f"Batch Size     : 256")
        print(f"Loss Fn        : FF Softplus (layer-wise)")
        print(f"Normalization  : L2 inter-layer (fixed)")
        print(f"Neg Label Fn   : Offset-based (fixed)")
        print()

        for epoch in range(1, epochs + 1):
            print_diagnostic_block(epoch, epoch_store[epoch], train_data.classes)

        # ───────── FINAL RESULT ─────────
        best_epoch = max(epoch_store, key=lambda e: epoch_store[e]['acc'])
        best       = epoch_store[best_epoch]
        final      = epoch_store[epochs]
        total_time = sum(epoch_times)

        print("\n" + "=" * 65)
        print("  FINAL RESULT")
        print("=" * 65)

        print(f"\n  --- Best Epoch ---")
        print(f"  Epoch          : {best_epoch}")
        print(f"  Train Acc      : {best['train_acc']:.2f}%")
        print(f"  Test Acc       : {best['acc']:.2f}%")
        print(f"  Loss           : {best['loss']:.2f}")

        print(f"\n  --- Final Epoch ({epochs}) Summary ---")
        print(f"  Final Test Acc : {final['acc']:.2f}%")
        print(f"  Final Loss     : {final['loss']:.2f}")
        print(f"  Total Time     : {total_time:.1f}s  ({total_time/epochs:.1f}s / epoch avg)")

        print(f"\n  --- Final Per-class Accuracy ---")
        for i, name in enumerate(train_data.classes):
            c, t = final['per_class'][i]
            print(f"    {name:<12}: {100*c/max(t,1):.2f}%  ({c}/{t})")

        print(f"\n  --- Final Confusion Matrix ---")
        print_confusion(final['cm'], train_data.classes)

        print("=" * 65)

        # ───────── VISUALISATION ─────────
        print("\n  Generating thesis plots...")
        plot_all(epoch_store, train_data.classes, epochs, best_epoch)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
net = FFNet()
net.train(train_loader, epochs)
