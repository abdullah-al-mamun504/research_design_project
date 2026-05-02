# ################################################################################
# BP vs FF Comparison Experiment
# Supports:
#   Experiment 1 — Simple Shapes (4 variants)
#   Experiment 2 — MNIST / FashionMNIST / EMNIST
#
# Flow:
#   1. Choose experiment + dataset
#   2. Set hyperparameters + trials (2–5)
#   3. All BP trials → All FF trials (live epoch lines)
#   4. Per-trial summaries
#   5. Full diagnostic dump (all epochs, all trials)
#   6. Cross-trial comparison + optional statistical tests
#   7. Per-trial plots (BP then FF)
#   8. Comparison plots
#   9. Optional feature analysis
#  10. Numbered menu to re-display any figure
# ################################################################################

import time, warnings, os, zipfile, itertools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

if IN_COLAB:
    drive.mount('/content/drive')
    BASE_DRIVE = "/content/drive/MyDrive/ds_research_design"
    BASE_LOCAL = "/content/dataset"
else:
    BASE_LOCAL = r"D:\dataset"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH  = 256
SEEDS  = [42, 123, 456, 789, 1011]

print(f"  Running on: {DEVICE.upper()}")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET EXTRACTION (Colab only) — fixes nested directory issue
# Extract zip directly into BASE_LOCAL so paths are flat:
#   BASE_LOCAL/simple_shapes_centered/train/...  (correct)
# NOT:
#   BASE_LOCAL/simple_shapes_centered/simple_shapes_centered/train/...  (wrong)
# ─────────────────────────────────────────────────────────────────────────────
if IN_COLAB:
    os.makedirs(BASE_LOCAL, exist_ok=True)
    ZIP_FILES = [
        "emnist_dataset.zip",
        "fashion_mnist_dataset.zip",
        "minist_dataset_experiment.zip",
        "simple_shapes_centered.zip",
        "simple_shapes_centered_noisy.zip",
        "simple_shapes_off_centered.zip",
        "simple_shapes_off_centered_noisy.zip",
    ]
    for z in ZIP_FILES:
        zip_path    = os.path.join(BASE_DRIVE, z)
        folder_name = z.replace(".zip", "")
        expected    = os.path.join(BASE_LOCAL, folder_name)
        if os.path.exists(expected):
            print(f"  Already extracted: {folder_name}")
            continue
        if not os.path.exists(zip_path):
            print(f"  Zip not found, skipping: {z}")
            continue
        print(f"  Extracting {z} → {BASE_LOCAL}")
        # Extract directly to BASE_LOCAL — zip contains folder_name/ at root
        # so result is BASE_LOCAL/folder_name/train/... (no double nesting)
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(BASE_LOCAL)
        print(f"    → {expected}")

# ─────────────────────────────────────────────────────────────────────────────
# EXPERIMENT SELECTION
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "="*70)
print("  BP vs FF Comparison Framework")
print("="*70)
print("  1: BP vs FF — Simple Shapes")
print("  2: BP vs FF — MNIST / FashionMNIST / EMNIST")
exp_choice = input("\nSelect experiment (1 / 2): ").strip()

# ─────────────────────────────────────────────────────────────────────────────
# DATASET SETUP
# ─────────────────────────────────────────────────────────────────────────────
_T_GRAY = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
_T_STD = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

if exp_choice == '1':
    EXP_NAME = "Simple Shapes"
    print("\n  1: Centered Shapes")
    print("  2: Not Centered Shapes")
    print("  3: Centered Noisy Shapes")
    print("  4: Not Centered Noisy Shapes")
    ds_choice = input("Select dataset (1–4): ").strip()
    _DS_MAP = {
        '1': ("simple_shapes_centered",          "Centered Shapes"),
        '2': ("simple_shapes_off_centered",       "Not Centered Shapes"),
        '3': ("simple_shapes_centered_noisy",     "Centered Noisy Shapes"),
        '4': ("simple_shapes_off_centered_noisy", "Not Centered Noisy Shapes"),
    }
    _folder, DATASET_NAME = _DS_MAP.get(ds_choice, _DS_MAP['1'])
    data_path   = os.path.join(BASE_LOCAL, _folder)
    NUM_CLASSES = 5

    print(f"  Path: {data_path}")
    print(f"  Train exists: {os.path.exists(os.path.join(data_path,'train'))}")
    print(f"  Test  exists: {os.path.exists(os.path.join(data_path,'test'))}")

    def _make_datasets(seed):
        torch.manual_seed(seed); np.random.seed(seed)
        tr = datasets.ImageFolder(os.path.join(data_path, "train"), transform=_T_GRAY)
        te = datasets.ImageFolder(os.path.join(data_path, "test"),  transform=_T_GRAY)
        return tr, te

else:
    EXP_NAME = "Standard Benchmark"
    print("\n  1: MNIST")
    print("  2: FashionMNIST")
    print("  3: EMNIST (balanced, 47 classes)")
    ds_choice = input("Select dataset (1 / 2 / 3): ").strip()

    if ds_choice == '1':
        DATASET_NAME = "MNIST"; NUM_CLASSES = 10
        _root = os.path.join(BASE_LOCAL, "minist_dataset_experiment")
        def _make_datasets(seed):
            torch.manual_seed(seed); np.random.seed(seed)
            return (datasets.MNIST(_root, train=True,  download=True, transform=_T_STD),
                    datasets.MNIST(_root, train=False, download=False, transform=_T_STD))
    elif ds_choice == '2':
        DATASET_NAME = "FashionMNIST"; NUM_CLASSES = 10
        _root = os.path.join(BASE_LOCAL, "fashion_mnist_dataset")
        def _make_datasets(seed):
            torch.manual_seed(seed); np.random.seed(seed)
            return (datasets.FashionMNIST(_root, train=True,  download=True, transform=_T_STD),
                    datasets.FashionMNIST(_root, train=False, download=False, transform=_T_STD))
    else:
        DATASET_NAME = "EMNIST (balanced)"; NUM_CLASSES = 47
        _root = os.path.join(BASE_LOCAL, "emnist_dataset")
        def _make_datasets(seed):
            torch.manual_seed(seed); np.random.seed(seed)
            return (datasets.EMNIST(_root, split='balanced', train=True,  download=True, transform=_T_STD),
                    datasets.EMNIST(_root, split='balanced', train=False, download=False, transform=_T_STD))

# ─────────────────────────────────────────────────────────────────────────────
# HYPERPARAMETERS
# ─────────────────────────────────────────────────────────────────────────────
print()
epochs    = int(input("Enter number of epochs: "))
lr        = float(input("Enter learning rate (e.g. 0.001): "))
threshold = float(input("Enter FF threshold (e.g. 1.0): "))
while True:
    n_trials = int(input("Enter number of trials (2–5): "))
    if 2 <= n_trials <= 5: break
    print("  Must be 2–5.")
ask_feature = input("Run feature representation analysis? (y/n): ").strip().lower() == 'y'

print(f"\n  {EXP_NAME} — {DATASET_NAME}")
print(f"  Epochs: {epochs} | LR: {lr} | Threshold: {threshold} | Trials: {n_trials}")
print(f"  Seeds: {SEEDS[:n_trials]} | Device: {DEVICE.upper()}")
if NUM_CLASSES == 47:
    print(f"  NOTE: EMNIST inference = {NUM_CLASSES} forward passes/sample — evaluation is slow.")

# ─────────────────────────────────────────────────────────────────────────────
# SHARED HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def set_seed(s):
    torch.manual_seed(s); np.random.seed(s)

def make_ff_input(x, y):
    x    = x.view(x.size(0), -1)
    y_oh = F.one_hot(y, NUM_CLASSES).float() * 2 - 1
    x    = x.clone(); x[:, :NUM_CLASSES] = y_oh
    return x

def neg_labels(y):
    return (y + torch.randint(1, NUM_CLASSES, y.shape)) % NUM_CLASSES

def count_dead(h):
    return (h.mean(dim=0) <= 0.0).sum().item()

def get_gnorm(layers_list):
    norms = []
    for layer in layers_list:
        ps = [p for p in layer.parameters() if p.grad is not None]
        norms.append(sum(p.grad.norm().item()**2 for p in ps)**0.5 if ps else 0.0)
    return norms

def get_layer_loss_contrib(model, x, y, criterion):
    x, y = x.to(DEVICE), y.to(DEVICE)
    xf   = x.view(x.size(0), -1)
    h1   = F.relu(model.l1(xf)); h2 = F.relu(model.l2(h1))
    h3   = F.relu(model.l3(h2)); out = model.cls(h3)
    loss = criterion(out, y)
    ns   = [h1.norm().item(), h2.norm().item(), h3.norm().item(), out.norm().item()]
    tot  = sum(ns) + 1e-8; lv = loss.item()
    return [lv * n / tot for n in ns], loss

def bp_eval(model, loader):
    model.eval(); correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y  = x.to(DEVICE), y.to(DEVICE)
            correct += (model(x).argmax(1) == y).sum().item(); total += y.size(0)
    model.train(); return 100.0 * correct / total

def bp_preds_all(model, loader):
    model.eval(); ap, ay = [], []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            ap.append(model(x).argmax(1).cpu()); ay.append(y)
    model.train()
    return torch.cat(ap).numpy(), torch.cat(ay).numpy()

def bp_per_class(model, loader):
    cr = [0]*NUM_CLASSES; ct = [0]*NUM_CLASSES
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).argmax(1)
            for c in range(NUM_CLASSES):
                m = (y==c); cr[c] += (p[m]==y[m]).sum().item(); ct[c] += m.sum().item()
    model.train(); return [(cr[c], ct[c]) for c in range(NUM_CLASSES)]

def bp_confusion(model, loader):
    mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            p = model(x).argmax(1)
            for t_, p_ in zip(y, p): mat[t_][p_] += 1
    model.train(); return mat

# ─────────────────────────────────────────────────────────────────────────────
# BP MODEL
# ─────────────────────────────────────────────────────────────────────────────
class BPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1  = nn.Linear(784, 500); self.l2 = nn.Linear(500, 500)
        self.l3  = nn.Linear(500, 500); self.cls = nn.Linear(500, NUM_CLASSES)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.cls(F.relu(self.l3(F.relu(self.l2(F.relu(self.l1(x)))))))

    def activations(self, x):
        x  = x.view(x.size(0), -1)
        h1 = F.relu(self.l1(x)); h2 = F.relu(self.l2(h1)); h3 = F.relu(self.l3(h2))
        return [h1, h2, h3]

# ─────────────────────────────────────────────────────────────────────────────
# BP TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_bp_trial(seed, tidx, tr_data, te_data):
    set_seed(seed)
    tr_ld = DataLoader(tr_data, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_data, batch_size=BATCH, shuffle=False)
    model     = BPNet().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    store = {}; prev_loss = None; times = []

    print(f"\n  [BP | Trial {tidx+1} | seed={seed}]")
    print(f"  {'Ep':>4} | {'Loss':>9} | {'ΔLoss':>9} | {'Train%':>7} | {'Test%':>7} | {'Time':>5} | ETA")
    print("  " + "─"*64)

    for ep in range(1, epochs+1):
        model.train(); t0 = time.time()
        total_loss = 0.0; correct = total = 0
        ll = [0.0]*4; lgn = [0.0]*4; ld = [0]*3; nb = 0

        for x, y in tr_ld:
            x, y = x.to(DEVICE), y.to(DEVICE)
            contrib, loss = get_layer_loss_contrib(model, x, y, criterion)
            for i in range(4): ll[i] += contrib[i]
            optimizer.zero_grad(); loss.backward()
            gn = get_gnorm([model.l1, model.l2, model.l3, model.cls])
            for i in range(4): lgn[i] += gn[i]
            optimizer.step()
            with torch.no_grad():
                acts = model.activations(x)
                for i in range(3): ld[i] += count_dead(acts[i])
            total_loss += loss.item()
            p = model(x).argmax(1); correct += (p==y).sum().item(); total += y.size(0)
            nb += 1

        t_ep = time.time() - t0; times.append(t_ep)
        nb        = max(nb, 1)
        train_acc = 100.0 * correct / total
        test_acc  = bp_eval(model, te_ld)
        cm        = bp_confusion(model, te_ld)
        pc        = bp_per_class(model, te_ld)
        avg_gn    = [lgn[i]/nb for i in range(4)]
        avg_dead  = [ld[i]//nb for i in range(3)]

        delta      = f"{total_loss-prev_loss:+.3f}" if prev_loss else "       —"
        prev_loss  = total_loss
        eta        = (sum(times)/len(times)) * (epochs - ep)
        tag        = " ← CONVERGED" if test_acc == 100.0 else ""

        store[ep]  = dict(loss=total_loss, train_acc=train_acc, acc=test_acc,
                          cm=cm, per_class=pc, layer_loss=ll,
                          avg_gnorm=avg_gn, avg_dead=avg_dead)
        print(f"  {ep:>4} | {total_loss:>9.4f} | {delta:>9} | {train_acc:>6.2f}% | "
              f"{test_acc:>6.2f}% | {t_ep:>4.1f}s | {eta:.1f}s{tag}")

    return store, model, te_ld

# ─────────────────────────────────────────────────────────────────────────────
# FF LAYER / NET
# ─────────────────────────────────────────────────────────────────────────────
class FFLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.opt    = torch.optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        return torch.relu(self.linear(x))

    def train_step(self, xp, xn):
        hp = self.forward(xp); hn = self.forward(xn)
        gp = (hp**2).mean(1);  gn_ = (hn**2).mean(1)
        loss = (F.softplus(-gp + threshold) + F.softplus(gn_ - threshold)).mean()
        self.opt.zero_grad(); loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(self.parameters(), 1e9)
        self.opt.step()
        hp_n = hp / (hp.norm(dim=1, keepdim=True) + 1e-8)
        hn_n = hn / (hn.norm(dim=1, keepdim=True) + 1e-8)
        return (hp_n.detach(), hn_n.detach(), loss.item(),
                gp.mean().item(), gn_.mean().item(), gnorm.item(),
                hp.detach(), hn.detach())


class FFNet:
    def __init__(self):
        self.layers = [FFLayer(784, 500), FFLayer(500, 500), FFLayer(500, 500)]

    def predict(self, x):
        x = x.view(x.size(0), -1); scores = []
        for c in range(NUM_CLASSES):
            xc = make_ff_input(x, torch.full((x.size(0),), c))
            h  = xc; g = 0
            for layer in self.layers:
                h = layer.forward(h); g += (h**2).mean(1)
                h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            scores.append(g.unsqueeze(1))
        return torch.cat(scores, 1).argmax(1)

    def evaluate(self, loader):
        correct = total = 0
        for x, y in loader:
            p = self.predict(x); correct += (p==y).sum().item(); total += y.size(0)
        return 100.0 * correct / total

    def preds_all(self, loader):
        ap, ay = [], []
        for x, y in loader:
            ap.append(self.predict(x)); ay.append(y)
        return torch.cat(ap).numpy(), torch.cat(ay).numpy()

    def per_class(self, loader):
        cr = [0]*NUM_CLASSES; ct = [0]*NUM_CLASSES
        for x, y in loader:
            p = self.predict(x)
            for c in range(NUM_CLASSES):
                m = (y==c); cr[c] += (p[m]==y[m]).sum().item(); ct[c] += m.sum().item()
        return [(cr[c], ct[c]) for c in range(NUM_CLASSES)]

    def confusion(self, loader):
        mat = torch.zeros(NUM_CLASSES, NUM_CLASSES, dtype=torch.long)
        for x, y in loader:
            p = self.predict(x)
            for t_, p_ in zip(y, p): mat[t_][p_] += 1
        return mat

    def get_activations(self, loader):
        all_h, all_y = [], []
        for x, y in loader:
            xf = x.view(x.size(0), -1); h = xf; parts = []
            for layer in self.layers:
                h = layer.forward(h); parts.append(h.detach())
                h = h / (h.norm(dim=1, keepdim=True) + 1e-8)
            all_h.append(torch.cat(parts, dim=1)); all_y.append(y)
        return torch.cat(all_h).numpy(), torch.cat(all_y).numpy()

# ─────────────────────────────────────────────────────────────────────────────
# FF TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def train_ff_trial(seed, tidx, tr_data, te_data):
    set_seed(seed)
    tr_ld = DataLoader(tr_data, batch_size=BATCH, shuffle=True)
    te_ld = DataLoader(te_data, batch_size=BATCH, shuffle=False)
    net   = FFNet(); store = {}; prev_loss = None; times = []

    print(f"\n  [FF | Trial {tidx+1} | seed={seed}]")
    print(f"  {'Ep':>4} | {'Loss':>9} | {'ΔLoss':>9} | {'Train%':>7} | {'Test%':>7} | {'Time':>5} | ETA")
    print("  " + "─"*64)

    for ep in range(1, epochs+1):
        t0 = time.time()
        total_loss = 0.0
        ll=[0.0]*3; lgp=[0.0]*3; lgn=[0.0]*3; lgn2=[0.0]*3
        ldead=[0]*3; lhp=[0]*3; lhn=[0]*3
        nb=0; last_x=last_y=None

        for x, y in tr_ld:
            last_x, last_y = x, y
            yn  = neg_labels(y)
            xp  = make_ff_input(x, y); xn = make_ff_input(x, yn)
            for i, layer in enumerate(net.layers):
                xp,xn,loss,gpm,gnm,gnorm,hpr,hnr = layer.train_step(xp,xn)
                total_loss+=loss; ll[i]+=loss
                lgp[i]+=gpm; lgn[i]+=gnm; lgn2[i]+=gnorm
                ldead[i]+=count_dead(hpr)
                gp_b=(hpr**2).mean(1); gn_b=(hnr**2).mean(1)
                lhp[i]+=(gp_b>threshold).sum().item()
                lhn[i]+=(gn_b<threshold).sum().item()
            nb+=1

        t_ep = time.time()-t0; times.append(t_ep); nb=max(nb,1)
        avg_gp   = [lgp[i]/nb    for i in range(3)]
        avg_gn   = [lgn[i]/nb    for i in range(3)]
        avg_gn2  = [lgn2[i]/nb   for i in range(3)]
        avg_dead = [ldead[i]//nb  for i in range(3)]
        gap      = [avg_gp[i]-avg_gn[i] for i in range(3)]
        tsamp    = len(tr_data)*nb+1e-8
        hit_p    = [100*lhp[i]/tsamp for i in range(3)]
        hit_n    = [100*lhn[i]/tsamp for i in range(3)]

        with torch.no_grad():
            tr_p  = net.predict(last_x)
            train_acc = 100.0*(tr_p==last_y).sum().item()/last_y.size(0)

        test_acc = net.evaluate(te_ld)
        cm       = net.confusion(te_ld)
        pc       = net.per_class(te_ld)

        gap_note = ("← near-perfect separation" if min(gap)>1.5 else
                    "← clear separation"        if min(gap)>0.8 else
                    "← gap small"               if max(gap)<0.3 else "")

        delta     = f"{total_loss-prev_loss:+.3f}" if prev_loss else "       —"
        prev_loss = total_loss
        eta       = (sum(times)/len(times))*(epochs-ep)
        tag       = " ← CONVERGED" if test_acc==100.0 else ""

        store[ep] = dict(loss=total_loss, train_acc=train_acc, acc=test_acc,
                         cm=cm, per_class=pc, layer_loss=ll,
                         avg_gp=avg_gp, avg_gn=avg_gn, gap=gap,
                         gap_note=gap_note, avg_gnorm=avg_gn2,
                         hit_p_pct=hit_p, hit_n_pct=hit_n, avg_dead=avg_dead)
        print(f"  {ep:>4} | {total_loss:>9.3f} | {delta:>9} | {train_acc:>6.2f}% | "
              f"{test_acc:>6.2f}% | {t_ep:>4.1f}s | {eta:.1f}s{tag}")

    return store, net, te_ld

# ─────────────────────────────────────────────────────────────────────────────
# PER-TRIAL SUMMARY (printed after each trial)
# ─────────────────────────────────────────────────────────────────────────────
def print_trial_summary(label, tidx, seed, store):
    fin   = store[epochs]
    best_e = max(store, key=lambda e: store[e]['acc'])
    best  = store[best_e]
    print(f"\n  ┌─ {label} Trial {tidx+1} Summary (seed={seed}) " + "─"*28)
    print(f"  │  Best epoch  : {best_e}  | Test Acc: {best['acc']:.2f}%  | Loss: {best['loss']:.4f}")
    print(f"  │  Final epoch : {epochs}  | Test Acc: {fin['acc']:.2f}%  | Loss: {fin['loss']:.4f}")
    print(f"  │  Train Acc   : {fin['train_acc']:.2f}%")
    print(f"  └" + "─"*52)

# ─────────────────────────────────────────────────────────────────────────────
# FULL DIAGNOSTIC DUMP — printed after ALL trials complete
# ─────────────────────────────────────────────────────────────────────────────
def print_full_diagnostics(label, all_stores, seeds, class_names):
    print("\n" + "="*70)
    print(f"  FULL DIAGNOSTICS — {label}")
    print("="*70)
    for tidx, (store, seed) in enumerate(zip(all_stores, seeds)):
        print(f"\n  ── Trial {tidx+1} (seed={seed}) ──────────────────────────────")
        for ep in range(1, epochs+1):
            s  = store[ep]
            print(f"\n  [Epoch {ep}]  Loss: {s['loss']:.4f} | "
                  f"Train: {s['train_acc']:.2f}% | Test: {s['acc']:.2f}%")

            if label == "BP":
                ll  = s['layer_loss']; gn = s['avg_gnorm']; d = s['avg_dead']
                print(f"    Layer losses → L1:{ll[0]:>7.4f}  L2:{ll[1]:>7.4f}  "
                      f"L3:{ll[2]:>7.4f}  L4(cls):{ll[3]:>7.4f}")
                print(f"    Grad norm    → L1:{gn[0]:>6.3f}  L2:{gn[1]:>6.3f}  "
                      f"L3:{gn[2]:>6.3f}  L4(cls):{gn[3]:>6.3f}")
                print(f"    Dead neurons → L1:{d[0]:>4d}   L2:{d[1]:>4d}   L3:{d[2]:>4d}")
            else:
                ll = s['layer_loss']; gp = s['avg_gp']; gn_ = s['avg_gn']
                gap= s['gap'];        gn2= s['avg_gnorm']; d = s['avg_dead']
                hp = s['hit_p_pct']; hn = s['hit_n_pct']
                note = s['gap_note']
                conv = " ← CONVERGED" if s['acc']==100.0 else ""
                print(f"    Layer losses → L1:{ll[0]:>7.3f}  L2:{ll[1]:>7.3f}  L3:{ll[2]:>7.3f}")
                print(f"    Goodness g+  → L1:{gp[0]:>6.3f}  L2:{gp[1]:>6.3f}  L3:{gp[2]:>6.3f}")
                print(f"    Goodness g-  → L1:{gn_[0]:>6.3f}  L2:{gn_[1]:>6.3f}  L3:{gn_[2]:>6.3f}")
                print(f"    Goodness gap → L1:{gap[0]:>6.3f}  L2:{gap[1]:>6.3f}  "
                      f"L3:{gap[2]:>6.3f}   {note}{conv}")
                print(f"    Grad norm    → L1:{gn2[0]:>6.3f}  L2:{gn2[1]:>6.3f}  L3:{gn2[2]:>6.3f}")
                print(f"    Threshold hit→ g+:{hp[0]:>5.1f}%   g-:{hn[0]:>5.1f}%  (L1 shown)")
                print(f"    Dead neurons → L1:{d[0]:>4d}   L2:{d[1]:>4d}   L3:{d[2]:>4d}")

        # per-class accuracy (final epoch)
        fin  = store[epochs]
        pc   = fin['per_class']
        print(f"\n  Per-class accuracy (final epoch, trial {tidx+1}):")
        for i, name in enumerate(class_names):
            c, t = pc[i]
            print(f"    {name:<16}: {100*c/max(t,1):>6.2f}%  ({c}/{t})")

        # confusion matrix text
        cm = fin['cm']
        if NUM_CLASSES <= 15:
            print(f"\n  Confusion matrix (final epoch, trial {tidx+1}):")
            col_w = 8
            hdr   = f"  {'':16}" + "".join(f"{n:>{col_w}}" for n in class_names)
            print(hdr)
            print("  " + "─" * (len(hdr)-2))
            for i, row in enumerate(cm):
                print(f"  {class_names[i]:<16}" + "".join(f"{v.item():>{col_w}}" for v in row))
        else:
            print(f"\n  [Confusion matrix text not feasible for {NUM_CLASSES} classes — see Figure 5]")

# ─────────────────────────────────────────────────────────────────────────────
# CROSS-TRIAL COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
def compute_f1(preds, labels):
    return f1_score(labels, preds, average=None,
                    labels=list(range(NUM_CLASSES)), zero_division=0)

def print_comparison(bp_stores, ff_stores, bp_preds, ff_preds, true_labs, class_names):
    print("\n" + "="*70)
    print("  CROSS-TRIAL COMPARISON")
    print("="*70)

    bp_accs = [s[epochs]['acc'] for s in bp_stores]
    ff_accs = [s[epochs]['acc'] for s in ff_stores]
    print(f"\n  Final Test Accuracy (mean ± std):")
    print(f"    BP: {np.mean(bp_accs):.2f}% ± {np.std(bp_accs):.2f}%  "
          f"[{min(bp_accs):.2f}, {max(bp_accs):.2f}]")
    print(f"    FF: {np.mean(ff_accs):.2f}% ± {np.std(ff_accs):.2f}%  "
          f"[{min(ff_accs):.2f}, {max(ff_accs):.2f}]")
    winner = "BP" if np.mean(bp_accs) > np.mean(ff_accs) else "FF"
    print(f"    → {winner} leads by {abs(np.mean(bp_accs)-np.mean(ff_accs)):.2f}pp")

    bp_f1t = [compute_f1(p,t) for p,t in zip(bp_preds, true_labs)]
    ff_f1t = [compute_f1(p,t) for p,t in zip(ff_preds, true_labs)]
    bp_f1  = np.mean(bp_f1t, axis=0)
    ff_f1  = np.mean(ff_f1t, axis=0)

    print(f"\n  Per-class F1 (mean across trials, final epoch):")
    max_c = min(NUM_CLASSES, 20)
    print(f"  {'Class':<18} {'BP F1':>8} {'FF F1':>8} {'Δ':>8}")
    print("  " + "─"*44)
    for i in range(max_c):
        d    = bp_f1[i] - ff_f1[i]
        flag = " ▲BP" if d>0.03 else (" ▲FF" if d<-0.03 else "")
        print(f"  {class_names[i]:<18} {bp_f1[i]:>8.3f} {ff_f1[i]:>8.3f} {d:>+8.3f}{flag}")
    if NUM_CLASSES > 20:
        print(f"  ... ({NUM_CLASSES-20} more classes — see Figure 4)")

    bp_loss = [s[epochs]['loss'] for s in bp_stores]
    ff_loss = [s[epochs]['loss'] for s in ff_stores]
    print(f"\n  Final Loss (mean ± std):")
    print(f"    BP: {np.mean(bp_loss):.4f} ± {np.std(bp_loss):.4f}")
    print(f"    FF: {np.mean(ff_loss):.4f} ± {np.std(ff_loss):.4f}")

    run_stat = input("\n  Run statistical tests? (y/n): ").strip().lower() == 'y'
    if run_stat:
        t_stat, p_val = stats.ttest_rel(bp_accs, ff_accs)
        print(f"\n  Paired t-test (accuracy):")
        print(f"    t={t_stat:.4f}  p={p_val:.4f}  "
              f"{'Significant (p<0.05)' if p_val<0.05 else 'Not significant'}")
        print(f"\n  McNemar test (error patterns, per trial):")
        for i in range(n_trials):
            bc = (bp_preds[i]==true_labs[i]).astype(int)
            fc = (ff_preds[i]==true_labs[i]).astype(int)
            b  = ((bc==1)&(fc==0)).sum()
            c  = ((bc==0)&(fc==1)).sum()
            if b+c > 0:
                chi2 = (abs(b-c)-1)**2/(b+c)
                pmc  = 1 - stats.chi2.cdf(chi2, df=1)
                print(f"    Trial {i+1}: b={b} c={c} χ²={chi2:.3f} p={pmc:.4f} "
                      f"{'Sig' if pmc<0.05 else 'NS'}")
            else:
                print(f"    Trial {i+1}: b={b} c={c} → no discordant pairs")

    return bp_f1, ff_f1

# ─────────────────────────────────────────────────────────────────────────────
# PLOT HELPERS
# ─────────────────────────────────────────────────────────────────────────────
LC4 = ['#E63946','#2A9D8F','#E9C46A','#8338EC']
LC3 = LC4[:3]
C_BP, C_FF = '#2E86AB', '#E63946'

def _rc():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,
                         'axes.titlesize':11,'axes.titleweight':'bold',
                         'axes.spines.top':False,'axes.spines.right':False,
                         'figure.dpi':130})

def _class_labels(class_names):
    """Short labels for axes — truncate to 8 chars for readability."""
    return [n[:8] for n in class_names]

def _bar_classes(ax, vals, class_names, title, ylabel, colors):
    """Bar chart that adapts to number of classes."""
    xlabels = _class_labels(class_names)
    if NUM_CLASSES <= 20:
        ax.bar(np.arange(NUM_CLASSES), vals, color=colors, edgecolor='white', lw=.7)
        ax.set_xticks(np.arange(NUM_CLASSES))
        ax.set_xticklabels(xlabels, rotation=45, ha='right',
                            fontsize=5 if NUM_CLASSES>10 else 8)
    else:
        # top-15 and bottom-15 for readability
        idx_top = np.argsort(vals)[-15:][::-1]
        idx_bot = np.argsort(vals)[:15]
        for sub_ax, idx, sub_title in [(ax, idx_top, f"{title} (top-15)"),]:
            sub_ax.bar(np.arange(len(idx)), [vals[i] for i in idx],
                       color=[colors[i] for i in idx], edgecolor='white', lw=.7)
            sub_ax.set_xticks(np.arange(len(idx)))
            sub_ax.set_xticklabels([xlabels[i] for i in idx],
                                    rotation=45, ha='right', fontsize=6)
            sub_ax.set_title(sub_title)
    ax.set_title(title); ax.set_ylabel(ylabel)

def _cm_plot(ax, cm_np, class_names, title):
    """Confusion matrix heatmap — adapts to class count."""
    im = ax.imshow(cm_np, interpolation='nearest', cmap='Blues')
    plt.colorbar(im, ax=ax, fraction=.046, pad=.04)
    tfs = 4 if NUM_CLASSES > 20 else (6 if NUM_CLASSES > 10 else 8)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    lbl = _class_labels(class_names)
    ax.set_xticklabels(lbl, rotation=90, ha='right', fontsize=tfs)
    ax.set_yticklabels(lbl, fontsize=tfs)
    ax.set_title(title); ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    if NUM_CLASSES <= 12:
        th = cm_np.max()/2
        for i, j in itertools.product(range(NUM_CLASSES), range(NUM_CLASSES)):
            ax.text(j, i, str(cm_np[i,j]), ha='center', va='center', fontsize=6,
                    color='white' if cm_np[i,j]>th else 'black')

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 2 — Mean test accuracy curves (BP vs FF) with shaded std
# ─────────────────────────────────────────────────────────────────────────────
def fig2_accuracy_curves(bp_stores, ff_stores):
    _rc(); ep = list(range(1, epochs+1))
    bp_acc = np.array([[s[e]['acc'] for e in ep] for s in bp_stores])
    ff_acc = np.array([[s[e]['acc'] for e in ep] for s in ff_stores])
    bp_los = np.array([[s[e]['loss'] for e in ep] for s in bp_stores])
    ff_los = np.array([[s[e]['loss'] for e in ep] for s in ff_stores])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Figure 2 — Learning Curves | {DATASET_NAME} | "
                 f"{n_trials} trials | Epochs={epochs}", fontweight='bold')

    for ax, bm, bs, fm, fs, ylabel, title in [
        (axes[0], bp_acc.mean(0), bp_acc.std(0), ff_acc.mean(0), ff_acc.std(0),
         "Accuracy (%)", "Test Accuracy (mean ± std)"),
        (axes[1], bp_los.mean(0), bp_los.std(0), ff_los.mean(0), ff_los.std(0),
         "Loss", "Loss (mean ± std)"),
    ]:
        ax.plot(ep, bm, C_BP, lw=2.5, marker='o', ms=5, label='BP')
        ax.fill_between(ep, bm-bs, bm+bs, alpha=.2, color=C_BP)
        ax.plot(ep, fm, C_FF, lw=2.5, marker='s', ms=5, label='FF')
        ax.fill_between(ep, fm-fs, fm+fs, alpha=.2, color=C_FF)
        ax.set_title(title); ax.set_xlabel("Epoch"); ax.set_ylabel(ylabel)
        ax.legend(fontsize=9); ax.set_xticks(ep)

    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 3 — Box plot of final accuracies + per-trial scatter
# ─────────────────────────────────────────────────────────────────────────────
def fig3_boxplot(bp_stores, ff_stores):
    _rc()
    bp_fin = [s[epochs]['acc'] for s in bp_stores]
    ff_fin = [s[epochs]['acc'] for s in ff_stores]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Figure 3 — Final Accuracy Distribution | {DATASET_NAME}",
                 fontweight='bold')

    ax = axes[0]
    for pos, vals, col, label in [(1,bp_fin,C_BP,'BP'),(2,ff_fin,C_FF,'FF')]:
        ax.boxplot([vals], positions=[pos], widths=.4, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=.7),
                   medianprops=dict(color='black',lw=2))
        ax.scatter([pos]*len(vals), vals, color=col, zorder=5, s=60, edgecolors='white')
    ax.set_xticks([1,2]); ax.set_xticklabels(['BP','FF'])
    ax.set_title("Final Accuracy Distribution"); ax.set_ylabel("Acc (%)")

    ax = axes[1]
    tr = range(1, n_trials+1)
    ax.plot(tr, bp_fin, C_BP, lw=2, marker='o', ms=8, label='BP')
    ax.plot(tr, ff_fin, C_FF, lw=2, marker='s', ms=8, label='FF')
    ax.axhline(np.mean(bp_fin), color=C_BP, lw=1, ls='--', alpha=.5)
    ax.axhline(np.mean(ff_fin), color=C_FF, lw=1, ls='--', alpha=.5)
    ax.set_title("Final Accuracy per Trial"); ax.set_xlabel("Trial")
    ax.set_ylabel("Acc (%)"); ax.set_xticks(tr); ax.legend(fontsize=9)

    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 4 — Per-class F1 comparison + delta
# ─────────────────────────────────────────────────────────────────────────────
def fig4_f1(bp_f1, ff_f1, class_names):
    _rc()
    delta  = bp_f1 - ff_f1
    cmap   = plt.cm.tab20 if NUM_CLASSES > 10 else plt.cm.tab10
    colors = cmap(np.linspace(0,1,NUM_CLASSES))
    xlbl   = _class_labels(class_names)
    x      = np.arange(NUM_CLASSES); w = 0.38

    if NUM_CLASSES <= 20:
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Figure 4 — Per-class F1 | {DATASET_NAME}", fontweight='bold')
        ax = axes[0]
        ax.bar(x-w/2, bp_f1, w, label='BP', color=C_BP, alpha=.8, edgecolor='white')
        ax.bar(x+w/2, ff_f1, w, label='FF', color=C_FF, alpha=.8, edgecolor='white')
        ax.set_title("Per-class F1 (mean, final epoch)"); ax.set_ylabel("F1")
        ax.set_ylim(0,1.15); ax.set_xticks(x)
        ax.set_xticklabels(xlbl, rotation=45, ha='right', fontsize=8); ax.legend(fontsize=9)
        ax = axes[1]
        ax.bar(x, delta, color=[C_BP if d>=0 else C_FF for d in delta],
               edgecolor='white', lw=.5)
        ax.axhline(0, color='black', lw=.8)
        ax.set_title("F1 Δ (BP − FF) [+ = BP better]"); ax.set_ylabel("ΔF1")
        ax.set_xticks(x); ax.set_xticklabels(xlbl, rotation=45, ha='right', fontsize=8)
    else:
        # For 47 classes: top-15 and bottom-15 delta
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Figure 4 — Per-class F1 | {DATASET_NAME} | "
                     f"Top-15 & Bottom-15 by F1 delta", fontweight='bold')
        idx_top = np.argsort(delta)[-15:][::-1]
        idx_bot = np.argsort(delta)[:15]
        for ax, idx, sub in [(axes[0], idx_top, "Top-15 (BP leads most)"),
                              (axes[1], idx_bot, "Bottom-15 (FF leads most)")]:
            d_sub = delta[idx]
            ax.bar(np.arange(15), d_sub, color=[C_BP if d>=0 else C_FF for d in d_sub],
                   edgecolor='white', lw=.5)
            ax.axhline(0, color='black', lw=.8)
            ax.set_title(f"F1 Δ — {sub}"); ax.set_ylabel("ΔF1")
            ax.set_xticks(np.arange(15))
            ax.set_xticklabels([xlbl[i] for i in idx], rotation=45, ha='right', fontsize=7)

    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 5 — Confusion matrices side-by-side (best trial)
# ─────────────────────────────────────────────────────────────────────────────
def fig5_confusion(bp_stores, ff_stores, class_names):
    _rc()
    # Best trial = highest final test acc
    best_bp = int(np.argmax([s[epochs]['acc'] for s in bp_stores]))
    best_ff = int(np.argmax([s[epochs]['acc'] for s in ff_stores]))
    cm_bp   = bp_stores[best_bp][epochs]['cm'].numpy()
    cm_ff   = ff_stores[best_ff][epochs]['cm'].numpy()
    # normalised versions
    cm_bp_n = cm_bp.astype(float) / (cm_bp.sum(1, keepdims=True) + 1e-8)
    cm_ff_n = cm_ff.astype(float) / (cm_ff.sum(1, keepdims=True) + 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle(f"Figure 5 — Confusion Matrices | {DATASET_NAME}\n"
                 f"Best BP trial={best_bp+1}, Best FF trial={best_ff+1}",
                 fontweight='bold')

    _cm_plot(axes[0,0], cm_bp,   class_names, f"BP — Raw counts (trial {best_bp+1})")
    _cm_plot(axes[0,1], cm_ff,   class_names, f"FF — Raw counts (trial {best_ff+1})")
    _cm_plot(axes[1,0], cm_bp_n, class_names, f"BP — Normalised (trial {best_bp+1})")
    _cm_plot(axes[1,1], cm_ff_n, class_names, f"FF — Normalised (trial {best_ff+1})")

    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 6 — Loss convergence + diagnostics comparison
# ─────────────────────────────────────────────────────────────────────────────
def fig6_loss(bp_stores, ff_stores):
    _rc(); ep = list(range(1, epochs+1))
    bp_los = np.array([[s[e]['loss'] for e in ep] for s in bp_stores])
    ff_los = np.array([[s[e]['loss'] for e in ep] for s in ff_stores])
    bp_gap = np.array([[s[e]['train_acc']-s[e]['acc'] for e in ep] for s in bp_stores])
    ff_gap = np.array([[s[e]['train_acc']-s[e]['acc'] for e in ep] for s in ff_stores])
    bp_std = np.array([[s[e]['acc'] for e in ep] for s in bp_stores]).std(0)
    ff_std = np.array([[s[e]['acc'] for e in ep] for s in ff_stores]).std(0)

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle(f"Figure 6 — Loss & Training Dynamics | {DATASET_NAME}",
                 fontweight='bold')

    # Loss curves
    ax = axes[0,0]
    for ti, (s, col) in enumerate([(bp_los, C_BP), (ff_los, C_FF)]):
        lbl = 'BP' if col==C_BP else 'FF'
        ax.plot(ep, s.mean(0), col, lw=2.5, marker='o', ms=4, label=lbl)
        ax.fill_between(ep, s.mean(0)-s.std(0), s.mean(0)+s.std(0), alpha=.15, color=col)
    ax.set_title("Loss Convergence (mean ± std)"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss"); ax.legend(fontsize=9); ax.set_xticks(ep)

    # Delta loss per trial
    ax = axes[0,1]
    for ti, s in enumerate(bp_stores):
        ls_ = [s[e]['loss'] for e in ep]
        dl  = [0]+[ls_[i]-ls_[i-1] for i in range(1,len(ls_))]
        ax.plot(ep, dl, C_BP, lw=1, alpha=.5+.1*ti, label=f'BP T{ti+1}')
    for ti, s in enumerate(ff_stores):
        ls_ = [s[e]['loss'] for e in ep]
        dl  = [0]+[ls_[i]-ls_[i-1] for i in range(1,len(ls_))]
        ax.plot(ep, dl, C_FF, lw=1, ls='--', alpha=.5+.1*ti, label=f'FF T{ti+1}')
    ax.axhline(0, color='gray', lw=.8); ax.set_title("ΔLoss per Trial")
    ax.set_xlabel("Epoch"); ax.set_ylabel("ΔLoss"); ax.set_xticks(ep)

    # Train-Test gap
    ax = axes[1,0]
    ax.plot(ep, bp_gap.mean(0), C_BP, lw=2, marker='o', ms=4, label='BP gap')
    ax.fill_between(ep, bp_gap.mean(0)-bp_gap.std(0), bp_gap.mean(0)+bp_gap.std(0),
                    alpha=.15, color=C_BP)
    ax.plot(ep, ff_gap.mean(0), C_FF, lw=2, marker='s', ms=4, label='FF gap')
    ax.fill_between(ep, ff_gap.mean(0)-ff_gap.std(0), ff_gap.mean(0)+ff_gap.std(0),
                    alpha=.15, color=C_FF)
    ax.axhline(0, color='gray', ls='--', lw=1)
    ax.set_title("Train−Test Gap (mean ± std)"); ax.set_xlabel("Epoch")
    ax.set_ylabel("Gap (%)"); ax.legend(fontsize=9); ax.set_xticks(ep)

    # Accuracy std (stability)
    ax = axes[1,1]
    ax.plot(ep, bp_std, C_BP, lw=2, marker='o', ms=4, label='BP std')
    ax.plot(ep, ff_std, C_FF, lw=2, marker='s', ms=4, label='FF std')
    ax.set_title("Test Accuracy Std across Trials (Stability)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Std (%)"); ax.legend(fontsize=9)
    ax.set_xticks(ep)

    plt.tight_layout(); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# FIGURE 7 — PCA / t-SNE feature analysis
# ─────────────────────────────────────────────────────────────────────────────
def fig7_features(bp_feats, bp_labs, ff_feats, ff_labs, class_names):
    _rc()
    cmap_pts  = plt.cm.tab20 if NUM_CLASSES > 10 else plt.cm.tab10
    cols_pts  = cmap_pts(np.linspace(0, 1, NUM_CLASSES))
    MAX_PTS   = 2000
    idx_bp    = np.random.choice(len(bp_feats), min(MAX_PTS,len(bp_feats)), replace=False)
    idx_ff    = np.random.choice(len(ff_feats), min(MAX_PTS,len(ff_feats)), replace=False)
    bf, bl    = bp_feats[idx_bp], bp_labs[idx_bp]
    ff_s, fl  = ff_feats[idx_ff], ff_labs[idx_ff]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f"Figure 7 — Feature Representation Analysis | {DATASET_NAME}\n"
                 f"(Trial 1, test set, all 3 hidden layers concatenated)",
                 fontweight='bold', y=0.99)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.48, wspace=0.38)

    # PCA BP
    pca_bp = PCA(n_components=2).fit_transform(bf)
    ax     = fig.add_subplot(gs[0,0])
    for c in range(NUM_CLASSES):
        m = bl==c
        if m.any():
            ax.scatter(pca_bp[m,0], pca_bp[m,1], c=[cols_pts[c]], s=8, alpha=.6,
                       label=class_names[c][:8])
    ax.set_title("PCA — BP"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if NUM_CLASSES <= 10: ax.legend(fontsize=7, markerscale=2)

    # PCA FF
    pca_ff = PCA(n_components=2).fit_transform(ff_s)
    ax     = fig.add_subplot(gs[0,1])
    for c in range(NUM_CLASSES):
        m = fl==c
        if m.any():
            ax.scatter(pca_ff[m,0], pca_ff[m,1], c=[cols_pts[c]], s=8, alpha=.6,
                       label=class_names[c][:8])
    ax.set_title("PCA — FF"); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    if NUM_CLASSES <= 10: ax.legend(fontsize=7, markerscale=2)

    # PCA cumulative variance
    ax    = fig.add_subplot(gs[0,2])
    nc    = min(20, bf.shape[1])
    ev_bp = PCA(n_components=nc).fit(bf).explained_variance_ratio_
    ev_ff = PCA(n_components=nc).fit(ff_s).explained_variance_ratio_
    ax.plot(range(1,nc+1), np.cumsum(ev_bp), C_BP, lw=2, marker='o', ms=4, label='BP')
    ax.plot(range(1,nc+1), np.cumsum(ev_ff), C_FF, lw=2, marker='s', ms=4, label='FF')
    ax.axhline(0.9, color='gray', ls='--', lw=1, label='90%')
    ax.set_title("PCA Cumulative Variance"); ax.set_xlabel("Components")
    ax.set_ylabel("Cum. Var."); ax.legend(fontsize=9)

    # t-SNE BP
    print("  Running t-SNE BP...")
    tsne_bp = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500).fit_transform(bf)
    ax      = fig.add_subplot(gs[1,0])
    for c in range(NUM_CLASSES):
        m = bl==c
        if m.any(): ax.scatter(tsne_bp[m,0], tsne_bp[m,1], c=[cols_pts[c]], s=8, alpha=.6)
    ax.set_title("t-SNE — BP"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    # t-SNE FF
    print("  Running t-SNE FF...")
    tsne_ff = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500).fit_transform(ff_s)
    ax      = fig.add_subplot(gs[1,1])
    for c in range(NUM_CLASSES):
        m = fl==c
        if m.any(): ax.scatter(tsne_ff[m,0], tsne_ff[m,1], c=[cols_pts[c]], s=8, alpha=.6)
    ax.set_title("t-SNE — FF"); ax.set_xlabel("dim 1"); ax.set_ylabel("dim 2")

    # Sparsity
    def layer_sparsity(F_):
        return [(F_[:,i*500:(i+1)*500]<=0).mean() for i in range(3)]
    sp_bp = layer_sparsity(bf); sp_ff = layer_sparsity(ff_s)
    ax    = fig.add_subplot(gs[1,2])
    xp    = np.arange(3); w = 0.38
    ax.bar(xp-w/2, sp_bp, w, label='BP', color=C_BP, alpha=.8, edgecolor='white')
    ax.bar(xp+w/2, sp_ff, w, label='FF', color=C_FF, alpha=.8, edgecolor='white')
    ax.set_title("Sparsity (fraction inactive)"); ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction"); ax.set_ylim(0,1)
    ax.set_xticks(xp); ax.set_xticklabels(['L1','L2','L3']); ax.legend(fontsize=9)

    # Entropy
    def layer_entropy(F_):
        ent = []
        for i in range(3):
            ch = np.clip(F_[:,i*500:(i+1)*500], 0, None)
            ma = ch.mean(0)+1e-10; ma /= ma.sum()
            ent.append(-np.sum(ma*np.log(ma+1e-10)))
        return ent
    en_bp = layer_entropy(bf); en_ff = layer_entropy(ff_s)
    ax    = fig.add_subplot(gs[2,0])
    ax.bar(xp-w/2, en_bp, w, label='BP', color=C_BP, alpha=.8, edgecolor='white')
    ax.bar(xp+w/2, en_ff, w, label='FF', color=C_FF, alpha=.8, edgecolor='white')
    ax.set_title("Activation Entropy"); ax.set_xlabel("Layer"); ax.set_ylabel("Entropy (nats)")
    ax.set_xticks(xp); ax.set_xticklabels(['L1','L2','L3']); ax.legend(fontsize=9)

    # Intrinsic dimensionality
    def intrinsic_dim(F_):
        from sklearn.neighbors import NearestNeighbors
        id_ = []
        for i in range(3):
            ch = F_[:,i*500:(i+1)*500]
            nn = NearestNeighbors(n_neighbors=11).fit(ch)
            d, _= nn.kneighbors(ch)
            mu  = d[:,2]/(d[:,1]+1e-10); mu = mu[mu>1]
            id_.append(1.0/(np.log(mu).mean()+1e-10) if len(mu)>0 else float('nan'))
        return id_
    print("  Computing intrinsic dimensionality...")
    id_bp = intrinsic_dim(bf); id_ff = intrinsic_dim(ff_s)
    ax    = fig.add_subplot(gs[2,1])
    ax.plot([1,2,3], id_bp, C_BP, lw=2, marker='o', ms=7, label='BP')
    ax.plot([1,2,3], id_ff, C_FF, lw=2, marker='s', ms=7, label='FF')
    ax.set_title("Intrinsic Dimensionality (2-NN)"); ax.set_xlabel("Layer")
    ax.set_ylabel("Est. ID"); ax.set_xticks([1,2,3])
    ax.set_xticklabels(['L1','L2','L3']); ax.legend(fontsize=9)

    # Mean activation magnitude
    def layer_mag(F_):
        return [F_[:,i*500:(i+1)*500].mean() for i in range(3)]
    mg_bp = layer_mag(bf); mg_ff = layer_mag(ff_s)
    ax    = fig.add_subplot(gs[2,2])
    ax.bar(xp-w/2, mg_bp, w, label='BP', color=C_BP, alpha=.8, edgecolor='white')
    ax.bar(xp+w/2, mg_ff, w, label='FF', color=C_FF, alpha=.8, edgecolor='white')
    ax.set_title("Mean Activation Magnitude"); ax.set_xlabel("Layer"); ax.set_ylabel("Mean")
    ax.set_xticks(xp); ax.set_xticklabels(['L1','L2','L3']); ax.legend(fontsize=9)

    plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# PER-TRIAL FIGURES (BP and FF individual diagnostics)
# ─────────────────────────────────────────────────────────────────────────────
def plot_bp_trial(store, tidx, seed):
    _rc(); ep = list(range(1, epochs+1))
    losses = [store[e]['loss'] for e in ep]
    tr_acc = [store[e]['train_acc'] for e in ep]
    te_acc = [store[e]['acc'] for e in ep]
    best_e = max(store, key=lambda e: store[e]['acc'])

    fig = plt.figure(figsize=(18,12))
    fig.suptitle(f"BP — {DATASET_NAME} | Trial {tidx+1} | seed={seed} | "
                 f"LR={lr} | Epochs={epochs}", fontsize=12, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.48,wspace=0.38)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(ep,losses,C_BP,lw=2,marker='o',ms=4)
    ax.axvline(best_e,color='gray',ls='--',lw=1,alpha=.7,label=f'Best ep {best_e}')
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[0,1])
    ax.plot(ep,tr_acc,'#2A9D8F',lw=2,marker='o',ms=4,label='Train')
    ax.plot(ep,te_acc,C_FF,lw=2,marker='s',ms=4,label='Test')
    ax.axvline(best_e,color='gray',ls='--',lw=1,alpha=.7)
    ax.set_title("Train vs Test Acc"); ax.set_ylim(0,105)
    ax.legend(fontsize=8); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[0,2])
    for l in range(4):
        ax.plot(ep,[store[e]['layer_loss'][l] for e in ep],LC4[l],lw=2,marker='o',ms=3,
                label=['L1','L2','L3','L4(cls)'][l])
    ax.set_title("Layer-wise Loss"); ax.legend(fontsize=7); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[1,0])
    for l in range(3):
        ax.plot(ep,[store[e]['avg_dead'][l] for e in ep],LC4[l],lw=2,marker='o',ms=3,
                label=f'L{l+1}')
    ax.set_title("Dead Neurons"); ax.legend(fontsize=8); ax.set_xticks(ep)

    ax  = fig.add_subplot(gs[1,1])
    pc  = [100*store[epochs]['per_class'][i][0]/max(store[epochs]['per_class'][i][1],1)
           for i in range(NUM_CLASSES)]
    cmap= plt.cm.tab20 if NUM_CLASSES>10 else plt.cm.tab10
    _bar_classes(ax, pc, _last_class_names,
                 f"Per-class Acc (Ep {epochs})", "Acc (%)",
                 cmap(np.linspace(0,1,NUM_CLASSES)))
    ax.set_ylim(0,115)

    ax = fig.add_subplot(gs[1,2])
    _cm_plot(ax, store[epochs]['cm'].numpy(), _last_class_names,
             f"Confusion Matrix (Ep {epochs})")

    ax = fig.add_subplot(gs[2,0])
    for l in range(4):
        ax.plot(ep,[store[e]['avg_gnorm'][l] for e in ep],LC4[l],lw=2,marker='o',ms=3,
                label=['L1','L2','L3','L4'][l])
    ax.set_title("Grad Norm"); ax.legend(fontsize=7); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[2,1])
    dl = [0]+[losses[i]-losses[i-1] for i in range(1,len(losses))]
    ax.bar(ep,dl,color=[C_FF if d>=0 else '#2A9D8F' for d in dl],edgecolor='white',lw=.5)
    ax.axhline(0,color='black',lw=.8); ax.set_title("ΔLoss"); ax.set_xticks(ep)

    ax  = fig.add_subplot(gs[2,2])
    gap = [tr_acc[i]-te_acc[i] for i in range(len(ep))]
    ax.plot(ep,gap,'#F18F01',lw=2,marker='o',ms=4)
    ax.axhline(0,color='gray',ls='--',lw=1)
    ax.fill_between(ep,gap,0,where=[g>0 for g in gap],alpha=.15,color=C_FF,label='Overfit')
    ax.fill_between(ep,gap,0,where=[g<=0 for g in gap],alpha=.15,color='#2A9D8F',label='Underfit')
    ax.set_title("Train−Test Gap"); ax.legend(fontsize=8); ax.set_xticks(ep)

    plt.tight_layout(rect=[0,0,1,0.97]); plt.show()


def plot_ff_trial(store, tidx, seed):
    _rc(); ep = list(range(1, epochs+1))
    losses = [store[e]['loss'] for e in ep]
    tr_acc = [store[e]['train_acc'] for e in ep]
    te_acc = [store[e]['acc'] for e in ep]
    best_e = max(store, key=lambda e: store[e]['acc'])

    fig = plt.figure(figsize=(18,12))
    fig.suptitle(f"FF — {DATASET_NAME} | Trial {tidx+1} | seed={seed} | "
                 f"LR={lr} | θ={threshold} | Epochs={epochs}",
                 fontsize=12, fontweight='bold', y=0.99)
    gs  = gridspec.GridSpec(3,3,figure=fig,hspace=0.48,wspace=0.38)

    ax = fig.add_subplot(gs[0,0])
    ax.plot(ep,losses,C_BP,lw=2,marker='o',ms=4)
    ax.axvline(best_e,color='gray',ls='--',lw=1,alpha=.7,label=f'Best ep {best_e}')
    ax.set_title("Total Loss"); ax.set_xlabel("Epoch"); ax.legend(fontsize=8); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[0,1])
    ax.plot(ep,tr_acc,'#2A9D8F',lw=2,marker='o',ms=4,label='Train*')
    ax.plot(ep,te_acc,C_FF,lw=2,marker='s',ms=4,label='Test')
    ax.axvline(best_e,color='gray',ls='--',lw=1,alpha=.7)
    ax.set_title("Train vs Test Acc"); ax.set_ylim(0,105)
    ax.legend(fontsize=8); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[0,2])
    for l in range(3):
        gaps_ = [store[e]['gap'][l] for e in ep]
        notes_ = [store[e]['gap_note'] for e in ep]
        ax.plot(ep, gaps_, LC3[l], lw=2, marker='o', ms=3, label=f'L{l+1}')
    ax.axhline(0.3,color='gray',ls=':',lw=1,alpha=.6,label='0.3 (small)')
    ax.axhline(0.8,color='black',ls=':',lw=1,alpha=.6,label='0.8 (clear)')
    ax.axhline(1.5,color='purple',ls=':',lw=1,alpha=.5,label='1.5 (near-perfect)')
    ax.set_title("Goodness Gap g⁺−g⁻"); ax.legend(fontsize=7); ax.set_xticks(ep)

    ax = fig.add_subplot(gs[1,0])
    for l in range(3):
        ax.plot(ep,[store[e]['avg_dead'][l] for e in ep],LC3[l],lw=2,marker='o',ms=3,
                label=f'L{l+1}')
    ax.set_title("Dead Neurons"); ax.legend(fontsize=8); ax.set_xticks(ep)

    ax  = fig.add_subplot(gs[1,1])
    pc  = [100*store[epochs]['per_class'][i][0]/max(store[epochs]['per_class'][i][1],1)
           for i in range(NUM_CLASSES)]
    cmap= plt.cm.tab20 if NUM_CLASSES>10 else plt.cm.tab10
    _bar_classes(ax, pc, _last_class_names,
                 f"Per-class Acc (Ep {epochs})", "Acc (%)",
                 cmap(np.linspace(0,1,NUM_CLASSES)))
    ax.set_ylim(0,115)

    ax = fig.add_subplot(gs[1,2])
    _cm_plot(ax, store[epochs]['cm'].numpy(), _last_class_names,
             f"Confusion Matrix (Ep {epochs})")

    ax = fig.add_subplot(gs[2,0])
    for l in range(3):
        ax.plot(ep,[store[e]['layer_loss'][l] for e in ep],LC3[l],lw=2,marker='o',ms=3,
                label=f'L{l+1}')
    ax.set_title("Layer-wise Loss"); ax.legend(fontsize=8); ax.set_xticks(ep)

    ax  = fig.add_subplot(gs[2,1])
    gp_ = store[epochs]['avg_gp']; gn_ = store[epochs]['avg_gn']
    xp_ = np.arange(3); w_ = 0.35
    ax.bar(xp_-w_/2, gp_, w_, label='g⁺', color='#2A9D8F', edgecolor='white')
    ax.bar(xp_+w_/2, gn_, w_, label='g⁻', color=C_FF, edgecolor='white')
    ax.axhline(threshold, color='black', ls='--', lw=1, label=f'θ={threshold}')
    ax.set_title(f"g⁺ vs g⁻ (Ep {epochs})"); ax.set_xlabel("Layer")
    ax.set_xticks(xp_); ax.set_xticklabels(['L1','L2','L3']); ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[2,2])
    for l in range(3):
        ax.plot(ep,[store[e]['avg_gnorm'][l] for e in ep],LC3[l],lw=2,marker='o',ms=3,
                label=f'L{l+1}')
    ax.set_title("Grad Norm"); ax.legend(fontsize=8); ax.set_xticks(ep)

    plt.tight_layout(rect=[0,0,1,0.97]); plt.show()

# ─────────────────────────────────────────────────────────────────────────────
# GET BP ACTIVATIONS
# ─────────────────────────────────────────────────────────────────────────────
def get_bp_activations(model, loader):
    all_h, all_y = [], []
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            acts = model.activations(x.to(DEVICE))
            all_h.append(torch.cat([a.cpu() for a in acts], dim=1)); all_y.append(y)
    model.train()
    return torch.cat(all_h).numpy(), torch.cat(all_y).numpy()

# ─────────────────────────────────────────────────────────────────────────────
# MAIN RUN
# ─────────────────────────────────────────────────────────────────────────────
bp_stores, ff_stores   = [], []
bp_preds,  ff_preds    = [], []
true_labs              = []
bp_models, ff_nets_all = [], []
_last_class_names      = None

# ══ PHASE 1: ALL BP TRIALS ══════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  PHASE 1 — BACKPROPAGATION ({n_trials} trials)")
print("="*70)

for ti in range(n_trials):
    tr_d, te_d        = _make_datasets(SEEDS[ti])
    _last_class_names = tr_d.classes
    te_ld             = DataLoader(te_d, batch_size=BATCH, shuffle=False)

    store, model, _   = train_bp_trial(SEEDS[ti], ti, tr_d, te_d)
    print_trial_summary("BP", ti, SEEDS[ti], store)

    p, l              = bp_preds_all(model, te_ld)
    bp_stores.append(store); bp_preds.append(p)
    true_labs.append(l);     bp_models.append(model)

# ══ PHASE 2: ALL FF TRIALS ══════════════════════════════════════════════════
print("\n" + "="*70)
print(f"  PHASE 2 — FORWARD-FORWARD ({n_trials} trials)")
print("="*70)

for ti in range(n_trials):
    tr_d, te_d  = _make_datasets(SEEDS[ti])
    te_ld       = DataLoader(te_d, batch_size=BATCH, shuffle=False)

    store, net, _ = train_ff_trial(SEEDS[ti], ti, tr_d, te_d)
    print_trial_summary("FF", ti, SEEDS[ti], store)

    p, _        = net.preds_all(te_ld)
    ff_stores.append(store); ff_preds.append(p); ff_nets_all.append(net)

# ══ PHASE 3: FULL DIAGNOSTIC DUMP ═══════════════════════════════════════════
print_full_diagnostics("BP", bp_stores, SEEDS[:n_trials], _last_class_names)
print_full_diagnostics("FF", ff_stores, SEEDS[:n_trials], _last_class_names)

# ══ PHASE 4: CROSS-TRIAL COMPARISON ══════════════════════════════════════════
bp_f1, ff_f1 = print_comparison(
    bp_stores, ff_stores, bp_preds, ff_preds, true_labs, _last_class_names)

# ══ PHASE 5: INDIVIDUAL TRIAL PLOTS ══════════════════════════════════════════
print("\n  Generating per-trial plots...")
for ti in range(n_trials):
    plot_bp_trial(bp_stores[ti], ti, SEEDS[ti])
    plot_ff_trial(ff_stores[ti], ti, SEEDS[ti])

# ══ PHASE 6: COMPARISON FIGURES ══════════════════════════════════════════════
print("  Generating comparison figures...")
fig2_accuracy_curves(bp_stores, ff_stores)
fig3_boxplot(bp_stores, ff_stores)
fig4_f1(bp_f1, ff_f1, _last_class_names)
fig5_confusion(bp_stores, ff_stores, _last_class_names)
fig6_loss(bp_stores, ff_stores)

# ══ PHASE 7: FEATURE ANALYSIS (optional) ═════════════════════════════════════
_bp_feats = _ff_feats = _bp_labs = _ff_labs = None
if ask_feature:
    print("\n  Running feature analysis (Trial 1)...")
    tr_d, te_d  = _make_datasets(SEEDS[0])
    feat_ld     = DataLoader(te_d, batch_size=BATCH, shuffle=False)
    _bp_feats, _bp_labs = get_bp_activations(bp_models[0], feat_ld)
    _ff_feats, _ff_labs = ff_nets_all[0].get_activations(feat_ld)
    fig7_features(_bp_feats, _bp_labs, _ff_feats, _ff_labs, _last_class_names)

# ══ NUMBERED MENU — re-display any figure at any time ════════════════════════
print("\n" + "="*70)
print("  FIGURE MENU — type a number to redisplay any figure")
print("="*70)
print("  1  → Mean accuracy curves (BP vs FF) with std bands")
print("  2  → Box plot of final accuracies")
print("  3  → Per-class F1 comparison")
print("  4  → Confusion matrices (raw + normalised, best trial)")
print("  5  → Loss convergence & training dynamics")
print("  6  → PCA / t-SNE feature analysis" +
      (" [requires feature analysis to have been run]" if not ask_feature else ""))
print("  bp1..bp5 → BP trial 1–5 individual plots")
print("  ff1..ff5 → FF trial 1–5 individual plots")
print("  q  → quit menu")

while True:
    choice = input("\n  Enter figure number (or q): ").strip().lower()
    if choice == 'q':
        break
    elif choice == '2':
        fig2_accuracy_curves(bp_stores, ff_stores)
    elif choice == '3':
        fig3_boxplot(bp_stores, ff_stores)
    elif choice == '4':
        fig4_f1(bp_f1, ff_f1, _last_class_names)
    elif choice == '5':
        fig5_confusion(bp_stores, ff_stores, _last_class_names)
    elif choice == '6':
        fig6_loss(bp_stores, ff_stores)
    elif choice == '7':
        if _bp_feats is None:
            print("  Feature analysis was not run. Re-run with ask_feature=True.")
        else:
            fig7_features(_bp_feats, _bp_labs, _ff_feats, _ff_labs, _last_class_names)
    elif choice.startswith('bp') and choice[2:].isdigit():
        ti = int(choice[2:])-1
        if 0 <= ti < n_trials: plot_bp_trial(bp_stores[ti], ti, SEEDS[ti])
        else: print(f"  Trial index out of range (1–{n_trials})")
    elif choice.startswith('ff') and choice[2:].isdigit():
        ti = int(choice[2:])-1
        if 0 <= ti < n_trials: plot_ff_trial(ff_stores[ti], ti, SEEDS[ti])
        else: print(f"  Trial index out of range (1–{n_trials})")
    else:
        print("  Unknown option. Try: 2, 3, 4, 5, 6, 7, bp1, ff2, q")

print("\n  ✓ Experiment complete.")
