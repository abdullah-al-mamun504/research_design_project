###########################################
# Experiment 3: FF Training — MNIST / FashionMNIST / EMNIST
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
# DATASET SELECTION
# ─────────────────────────────────────────
print("1: MNIST")
print("2: FashionMNIST")
print("3: EMNIST (balanced)")
choice = input("Select dataset (1 / 2 / 3): ").strip()

_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if choice == '1':
    data_path    = r"D:\Research design course\minist_dataset_experiment"
    dataset_name = "MNIST"
    NUM_CLASSES  = 10
    train_data   = datasets.MNIST(root=data_path, train=True,  download=False, transform=_transform)
    test_data    = datasets.MNIST(root=data_path, train=False, download=False, transform=_transform)
    print("--- Running Experiment: MNIST ---")

elif choice == '2':
    data_path    = r"D:\Research design course\fashion_mnist"
    dataset_name = "FashionMNIST"
    NUM_CLASSES  = 10
    train_data   = datasets.FashionMNIST(root=data_path, train=True,  download=False, transform=_transform)
    test_data    = datasets.FashionMNIST(root=data_path, train=False, download=False, transform=_transform)
    print("--- Running Experiment: FashionMNIST ---")

else:
    data_path    = r"D:\Research design course\emnist"
    dataset_name = "EMNIST (balanced)"
    NUM_CLASSES  = 47
    train_data   = datasets.EMNIST(root=data_path, split='balanced', train=True,  download=False, transform=_transform)
    test_data    = datasets.EMNIST(root=data_path, split='balanced', train=False, download=False, transform=_transform)
    print("--- Running Experiment: EMNIST (balanced) ---")

# ─────────────────────────────────────────
# HYPERPARAMETER INPUT
# ─────────────────────────────────────────
epochs    = int(input("Enter number of epochs: "))
lr        = float(input("Enter learning rate (e.g. 0.003): "))
threshold = float(input("Enter threshold (e.g. 1.0): "))

train_loader = DataLoader(train_data, batch_size=256, shuffle=True)
test_loader  = DataLoader(test_data,  batch_size=256, shuffle=False)

print(f"Train: {len(train_data)} | Test: {len(test_data)}")
print(f"Classes ({NUM_CLASSES}): {train_data.classes[:10]}{'...' if NUM_CLASSES > 10 else ''}")

# ─────────────────────────────────────────
# CORE HELPERS — logic unchanged
# ─────────────────────────────────────────
def make_input(x, y):
    """Embed one-hot label (±1) into first NUM_CLASSES pixels of flattened image."""
    x    = x.view(x.size(0), -1)
    y_oh = F.one_hot(y, NUM_CLASSES).float() * 2 - 1
    x    = x.clone()
    x[:, :NUM_CLASSES] = y_oh
    return x

def neg_labels(y):
    """Offset-based negative labels — guaranteed different from true label."""
    offsets = torch.randint(1, NUM_CLASSES, y.shape)
    return (y + offsets) % NUM_CLASSES

def count_dead_neurons(h):
    return (h.mean(dim=0) <= 0.0).sum().item()

# ─────────────────────────────────────────
# DIAGNOSTIC HELPERS — read-only
# ─────────────────────────────────────────
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

def confusion_matrix_ff(net, loader):
    matrix = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    for x, y in loader:
        preds = net.predict(x)
        for t, p in zip(y, preds):
            matrix[t][p] += 1
    return matrix

def print_confusion(matrix, class_names):
    # EMNIST 47×47 is unreadable in terminal — show in plot only
    if NUM_CLASSES > 15:
        print("  [Confusion matrix omitted in text — see plot]")
        return
    print("\n  Confusion matrix:")
    col_w = 7
    print(f"  {'':8}" + "".join(f"{n:>{col_w}}" for n in class_names))
    for i, row in enumerate(matrix):
        print(f"  {class_names[i]:<8}" + "".join(f"{v.item():>{col_w}}" for v in row))

def print_diagnostic_block(epoch, stored, class_names):
    print(f"\n[Epoch {epoch}]")
    print(f"  Loss: {stored['loss']:.2f} | "
          f"Train Acc: {stored['train_acc']:.2f}% | "
          f"Test Acc: {stored['acc']:.2f}%")

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
    print(f"  Goodness gap  →  L1: {g[0]:>6.3f}    L2: {g[1]:>6.3f}    L3: {g[2]:>6.3f}"
          f"   {stored['gap_note']}")
    print(f"  Grad norm     →  L1: {gn2[0]:>6.3f}    L2: {gn2[1]:>6.3f}    L3: {gn2[2]:>6.3f}")
    print(f"  Threshold hit →  g+: {hp[0]:>5.1f}%   g-: {hn[0]:>5.1f}%   (L1 shown)")
    print(f"  Dead neurons  →  L1: {d[0]:>4d}     L2: {d[1]:>4d}     L3: {d[2]:>4d}")

    print("\n  Per-class accuracy:")
    for i, name in enumerate(class_names):
        c, t = stored['per_class'][i]
        print(f"    {name:<14}: {100*c/max(t,1):.2f}%  ({c}/{t})")

    print_confusion(stored['cm'], class_names)

# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────
def plot_all(epoch_store, class_names, epochs, best_epoch):
    ep = list(range(1, epochs + 1))

    losses     = [epoch_store[e]['loss']      for e in ep]
    train_accs = [epoch_store[e]['train_acc'] for e in ep]
    test_accs  = [epoch_store[e]['acc']       for e in ep]
    gaps       = [[epoch_store[e]['gap'][l]   for e in ep] for l in range(3)]
    dead       = [[epoch_store[e]['avg_dead'][l] for e in ep] for l in range(3)]
    final_pc   = [100 * epoch_store[epochs]['per_class'][i][0] /
                  max(epoch_store[epochs]['per_class'][i][1], 1)
                  for i in range(NUM_CLASSES)]
    final_cm   = epoch_store[epochs]['cm'].numpy()

    layer_colors = ['#E63946', '#2A9D8F', '#E9C46A']

    plt.rcParams.update({
        'font.family':      'DejaVu Sans',
        'font.size':        11,
        'axes.titlesize':   13,
        'axes.titleweight': 'bold',
        'axes.spines.top':  False,
        'axes.spines.right':False,
        'figure.dpi':       150,
    })

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(
        f"Forward-Forward Network — {dataset_name}\n"
        f"Epochs: {epochs}  |  LR: {lr}  |  Threshold: {threshold}  |  "
        f"Classes: {NUM_CLASSES}  |  Train: {len(train_data)}  |  Test: {len(test_data)}",
        fontsize=14, fontweight='bold', y=0.98
    )

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.38)

    # 1. Loss curve
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(ep, losses, color='#2E86AB', linewidth=2, marker='o', markersize=5)
    ax1.axvline(best_epoch, color='gray', linestyle='--', linewidth=1,
                alpha=0.7, label=f'Best epoch {best_epoch}')
    ax1.set_title("Total Loss per Epoch")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9); ax1.set_xticks(ep)

    # 2. Train vs Test accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(ep, train_accs, color='#2A9D8F', linewidth=2, marker='o',
             markersize=5, label='Train')
    ax2.plot(ep, test_accs,  color='#E63946', linewidth=2, marker='s',
             markersize=5, label='Test')
    ax2.axvline(best_epoch, color='gray', linestyle='--', linewidth=1, alpha=0.7)
    ax2.set_title("Train vs Test Accuracy")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105); ax2.legend(fontsize=9); ax2.set_xticks(ep)

    # 3. Goodness gap per layer
    ax3 = fig.add_subplot(gs[0, 2])
    for l in range(3):
        ax3.plot(ep, gaps[l], color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax3.axhline(0.3, color='gray',  linestyle=':', linewidth=1, alpha=0.6, label='Gap=0.3')
    ax3.axhline(0.8, color='black', linestyle=':', linewidth=1, alpha=0.6, label='Gap=0.8')
    ax3.set_title("Goodness Gap per Layer\n(g⁺ − g⁻)")
    ax3.set_xlabel("Epoch"); ax3.set_ylabel("Gap")
    ax3.legend(fontsize=8); ax3.set_xticks(ep)

    # 4. Dead neurons
    ax4 = fig.add_subplot(gs[1, 0])
    for l in range(3):
        ax4.plot(ep, dead[l], color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax4.set_title("Dead Neurons per Layer")
    ax4.set_xlabel("Epoch"); ax4.set_ylabel("Dead Neuron Count")
    ax4.legend(fontsize=9); ax4.set_xticks(ep)

    # 5. Per-class accuracy bar chart
    ax5 = fig.add_subplot(gs[1, 1])
    x_pos  = np.arange(NUM_CLASSES)
    colors = plt.cm.tab20(np.linspace(0, 1, NUM_CLASSES)) if NUM_CLASSES > 10 \
             else plt.cm.tab10(np.linspace(0, 1, NUM_CLASSES))
    bars   = ax5.bar(x_pos, final_pc, color=colors, edgecolor='white', linewidth=0.8)
    if NUM_CLASSES <= 15:
        for bar, val in zip(bars, final_pc):
            ax5.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha='center', va='bottom',
                     fontsize=7, fontweight='bold')
    ax5.set_title(f"Per-class Accuracy\n(Final Epoch {epochs})")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_ylim(0, 115)
    ax5.set_xticks(x_pos)
    ax5.set_xticklabels(class_names, rotation=45, ha='right',
                        fontsize=6 if NUM_CLASSES > 15 else 8)

    # 6. Confusion matrix heatmap
    ax6 = fig.add_subplot(gs[1, 2])
    im = ax6.imshow(final_cm, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)
    tick_fs = 5 if NUM_CLASSES > 20 else 8
    ax6.set_xticks(range(NUM_CLASSES)); ax6.set_yticks(range(NUM_CLASSES))
    ax6.set_xticklabels(class_names, rotation=90, ha='right', fontsize=tick_fs)
    ax6.set_yticklabels(class_names, fontsize=tick_fs)
    ax6.set_title(f"Confusion Matrix\n(Final Epoch {epochs})")
    ax6.set_xlabel("Predicted"); ax6.set_ylabel("Actual")
    if NUM_CLASSES <= 15:
        thresh = final_cm.max() / 2.0
        for i in range(NUM_CLASSES):
            for j in range(NUM_CLASSES):
                ax6.text(j, i, str(final_cm[i, j]), ha='center', va='center',
                         fontsize=7,
                         color='white' if final_cm[i, j] > thresh else 'black')

    # 7. Layer-wise loss
    ax7 = fig.add_subplot(gs[2, 0])
    for l in range(3):
        ll_ep = [epoch_store[e]['layer_loss'][l] for e in ep]
        ax7.plot(ep, ll_ep, color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax7.set_title("Layer-wise Loss per Epoch")
    ax7.set_xlabel("Epoch"); ax7.set_ylabel("Loss")
    ax7.legend(fontsize=9); ax7.set_xticks(ep)

    # 8. Goodness g+ vs g- final epoch
    ax8 = fig.add_subplot(gs[2, 1])
    x_pos2  = np.arange(3)
    gp_vals = epoch_store[epochs]['avg_gp']
    gn_vals = epoch_store[epochs]['avg_gn']
    width   = 0.35
    ax8.bar(x_pos2 - width/2, gp_vals, width, label='g⁺ (positive)',
            color='#2A9D8F', edgecolor='white')
    ax8.bar(x_pos2 + width/2, gn_vals, width, label='g⁻ (negative)',
            color='#E63946', edgecolor='white')
    ax8.axhline(threshold, color='black', linestyle='--', linewidth=1,
                label=f'Threshold={threshold}')
    ax8.set_title(f"Goodness g⁺ vs g⁻ by Layer\n(Final Epoch {epochs})")
    ax8.set_xlabel("Layer"); ax8.set_ylabel("Goodness")
    ax8.set_xticks(x_pos2)
    ax8.set_xticklabels(['Layer 1', 'Layer 2', 'Layer 3'])
    ax8.legend(fontsize=9)

    # 9. Gradient norm per layer
    ax9 = fig.add_subplot(gs[2, 2])
    for l in range(3):
        gn_ep = [epoch_store[e]['avg_gnorm'][l] for e in ep]
        ax9.plot(ep, gn_ep, color=layer_colors[l], linewidth=2,
                 marker='o', markersize=5, label=f'Layer {l+1}')
    ax9.set_title("Gradient Norm per Layer")
    ax9.set_xlabel("Epoch"); ax9.set_ylabel("Grad Norm")
    ax9.legend(fontsize=9); ax9.set_xticks(ep)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

    # Uncomment to save:
    # fig.savefig(r"D:\results\ff_thesis_plots.png", dpi=300, bbox_inches='tight')

# ─────────────────────────────────────────
# FF LAYER — unchanged logic
# ─────────────────────────────────────────
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt    = torch.optim.Adam(self.parameters(), lr=lr)

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

        # L2 normalise before passing to next layer
        hp_norm = hp / (hp.norm(dim=1, keepdim=True) + 1e-8)
        hn_norm = hn / (hn.norm(dim=1, keepdim=True) + 1e-8)

        return (hp_norm.detach(), hn_norm.detach(), loss.item(),
                gp.mean().item(), gn.mean().item(), grad_norm.item(),
                hp.detach(), hn.detach())

# ─────────────────────────────────────────
# FF NET — NUM_CLASSES drives label embedding
# ─────────────────────────────────────────
class FFNet:
    def __init__(self):
        self.layers = [
            FFLayer(784, 500),
            FFLayer(500, 500),
            FFLayer(500, 500),
        ]

    def predict(self, x):
        """Try all NUM_CLASSES labels — pick highest cumulative goodness."""
        x = x.view(x.size(0), -1)
        scores = []
        for c in range(NUM_CLASSES):
            y_c = torch.full((x.size(0),), c)
            xc  = make_input(x, y_c)
            h, g = xc, 0
            for layer in self.layers:
                h  = layer.forward(h)
                g += (h ** 2).mean(dim=1)
                h  = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            scores.append(g.unsqueeze(1))
        return torch.cat(scores, dim=1).argmax(dim=1)

    def evaluate(self, loader):
        correct, total = 0, 0
        for x, y in loader:
            preds    = self.predict(x)
            correct += (preds == y).sum().item()
            total   += y.size(0)
        return 100.0 * correct / total

    def train(self, loader, epochs):
        epoch_store = {}
        prev_loss   = None
        epoch_times = []

        # ── header ──
        print("\n" + "=" * 70)
        print(f"  FORWARD-FORWARD EXPERIMENT — {dataset_name}")
        print(f"  Epochs: {epochs}  |  LR: {lr}  |  Threshold: {threshold}")
        print(f"  Classes: {NUM_CLASSES}  |  Train: {len(train_data)}  |  Test: {len(test_data)}")
        if NUM_CLASSES == 47:
            print(f"  NOTE: EMNIST has 47 classes — inference runs {NUM_CLASSES}× forward passes per sample.")
            print(f"        Expect slower evaluation than MNIST/FashionMNIST.")
        print("=" * 70)

        # ══════════════════════════════════════════
        # PHASE 1 — training: one line per epoch
        # ══════════════════════════════════════════
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
            last_x = last_y = None

            for x, y in loader:
                last_x, last_y = x, y          # track last batch for fast train acc
                y_neg = neg_labels(y)
                xp    = make_input(x, y)
                xn    = make_input(x, y_neg)

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

            total_samp = len(train_data) * nb + 1e-8
            hit_p_pct  = [100 * layer_hit_p[i] / total_samp for i in range(3)]
            hit_n_pct  = [100 * layer_hit_n[i] / total_samp for i in range(3)]

            gap_note = ("← gap small"        if max(gap) < 0.3 else
                        "← clear separation" if min(gap) > 0.8 else "")

            # fast train acc from last batch
            with torch.no_grad():
                train_preds = self.predict(last_x)
                train_acc   = 100.0 * (train_preds == last_y).sum().item() / last_y.size(0)

            test_acc  = self.evaluate(test_loader)
            cm        = confusion_matrix_ff(self, test_loader)
            per_class = per_class_accuracy(self, test_loader)

            delta_str     = f"{total_loss - prev_loss:+.2f}" if prev_loss is not None else "        —"
            prev_loss     = total_loss
            avg_time      = sum(epoch_times) / len(epoch_times)
            remaining     = avg_time * (epochs - epoch)
            converged_tag = " ← CONVERGED" if test_acc == 100.0 else ""

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

            print(f"Epoch {epoch:>3} | Loss: {total_loss:>8.2f} | ΔLoss: {delta_str:>9} | "
                  f"Train Acc: {train_acc:>6.2f}%* | Test Acc: {test_acc:>6.2f}% | "
                  f"Time: {epoch_time:>5.1f}s | ETA: {remaining:>5.1f}s{converged_tag}")

        total_time = sum(epoch_times)

        # ══════════════════════════════════════════
        # PHASE 2 — full diagnostics per epoch
        # ══════════════════════════════════════════
        print("\n" + "=" * 70)
        print("  EPOCH DIAGNOSTICS")
        print("=" * 70)
        print(f"  Dataset      : {dataset_name}")
        print(f"  Architecture : 784 → 500 → 500 → 500")
        print(f"  Optimizer    : Adam  |  Loss Fn: FF Softplus (layer-wise)  |  Batch: 256")
        print(f"  Neg Label    : Offset-based  |  Normalisation: L2 inter-layer")
        print(f"  (* Train Acc = last-batch approximation)")

        for epoch in range(1, epochs + 1):
            print_diagnostic_block(epoch, epoch_store[epoch], train_data.classes)

        # ══════════════════════════════════════════
        # PHASE 3 — final results
        # ══════════════════════════════════════════
        best_epoch = max(epoch_store, key=lambda e: epoch_store[e]['acc'])
        best       = epoch_store[best_epoch]
        final      = epoch_store[epochs]

        print("\n" + "=" * 70)
        print("  FINAL RESULTS")
        print("=" * 70)

        print(f"\n  --- Best Epoch ---")
        print(f"  Epoch          : {best_epoch}")
        print(f"  Train Acc      : {best['train_acc']:.2f}%*")
        print(f"  Test Acc       : {best['acc']:.2f}%")
        print(f"  Loss           : {best['loss']:.2f}")

        print(f"\n  --- Final Epoch ({epochs}) ---")
        print(f"  Test Acc       : {final['acc']:.2f}%")
        print(f"  Loss           : {final['loss']:.2f}")
        print(f"  Total Time     : {total_time:.1f}s  ({total_time/epochs:.1f}s / epoch avg)")

        print(f"\n  --- Per-class Accuracy (final epoch) ---")
        for i, name in enumerate(train_data.classes):
            c, t = final['per_class'][i]
            print(f"    {name:<14}: {100*c/max(t,1):.2f}%  ({c}/{t})")

        print(f"\n  --- Final Confusion Matrix ---")
        print_confusion(final['cm'], train_data.classes)

        print("=" * 70)

        # ══════════════════════════════════════════
        # PHASE 4 — plots
        # ══════════════════════════════════════════
        print("\n  Generating plots...")
        plot_all(epoch_store, train_data.classes, epochs, best_epoch)


# ─────────────────────────────────────────
# RUN
# ─────────────────────────────────────────
net = FFNet()
net.train(train_loader, epochs)
