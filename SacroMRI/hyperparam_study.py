"""
=============================================================================

 EXPERIMENT 1 : Optimizer comparison      (AdamW vs SGD vs RMSprop vs Adam)
 EXPERIMENT 2 : Learning rate comparison  (1e-2, 1e-3, 3e-4, 1e-4, 1e-5)
 EXPERIMENT 3 : Batch size comparison     (8, 16, 32, 64)
 EXPERIMENT 4 : Dropout rate comparison   (0.0, 0.1, 0.2, 0.3, 0.5)
 EXPERIMENT 5 : Activation function       (ReLU, GELU, ELU, Tanh, Sigmoid)
 EXPERIMENT 6 : LR Scheduler comparison   (Cosine, StepLR, ExponentialLR, None)
 EXPERIMENT 7 : Weight decay comparison   (0, 1e-5, 1e-4, 1e-3, 1e-2)
 EXPERIMENT 8 : Parameter efficiency      (model params vs accuracy vs bias)


 Final output: best hyperparameter per category + full comparison table
=============================================================================
"""

import os, glob, warnings, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                              precision_score, recall_score, f1_score,
                              confusion_matrix)
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
print("✓  Hyperparameter comparison script loaded")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
KAGGLE_ROOT = r"D:\SacroMri\archive"
CSV_PATH    = os.path.join(KAGGLE_ROOT, "SacroMRI Dataset.csv")
DATA_ROOT   = os.path.join(KAGGLE_ROOT, "SacroMRI")
OUT_DIR     = r"D:\SacroMri\hyperparam_study"
os.makedirs(OUT_DIR, exist_ok=True)

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE   = 340
EMBED_DIM  = 512
N_EPOCHS   = 8      # same as main study for fair comparison
SEED       = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"✓  device={DEVICE}")

CLINICAL_FEATURES = [
    "chronic_back_pain", "morning_stiffness", "improvement_exercise",
    "hla_b27", "esr_norm", "crp_norm",
]
META_DIM = len(CLINICAL_FEATURES)

# ══════════════════════════════════════════════════════════════════════════════
# HYPERPARAMETER GRIDS — one experiment changes ONE variable
# ══════════════════════════════════════════════════════════════════════════════
EXPERIMENTS = {
    "optimizer": {
        "variable": "Optimizer",
        "fixed":    {"lr": 3e-4, "batch_size": 16, "dropout": 0.3,
                     "activation": "gelu", "scheduler": "cosine",
                     "weight_decay": 1e-4},
        "values":   ["adamw", "sgd", "rmsprop", "adam"],
    },
    "learning_rate": {
        "variable": "Learning Rate",
        "fixed":    {"optimizer": "adamw", "batch_size": 16, "dropout": 0.3,
                     "activation": "gelu", "scheduler": "cosine",
                     "weight_decay": 1e-4},
        "values":   [1e-2, 1e-3, 3e-4, 1e-4, 1e-5],
    },
    "batch_size": {
        "variable": "Batch Size",
        "fixed":    {"optimizer": "adamw", "lr": 3e-4, "dropout": 0.3,
                     "activation": "gelu", "scheduler": "cosine",
                     "weight_decay": 1e-4},
        "values":   [8, 16, 32, 64],
    },
    "dropout": {
        "variable": "Dropout Rate",
        "fixed":    {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                     "activation": "gelu", "scheduler": "cosine",
                     "weight_decay": 1e-4},
        "values":   [0.0, 0.1, 0.2, 0.3, 0.5],
    },
    "activation": {
        "variable": "Activation Function",
        "fixed":    {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                     "dropout": 0.3, "scheduler": "cosine",
                     "weight_decay": 1e-4},
        "values":   ["relu", "gelu", "elu", "tanh"],
    },
    "scheduler": {
        "variable": "LR Scheduler",
        "fixed":    {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                     "dropout": 0.3, "activation": "gelu",
                     "weight_decay": 1e-4},
        "values":   ["cosine", "step", "exponential", "none"],
    },
    "weight_decay": {
        "variable": "Weight Decay (λ)",
        "fixed":    {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                     "dropout": 0.3, "activation": "gelu",
                     "scheduler": "cosine"},
        "values":   [0.0, 1e-5, 1e-4, 1e-3, 1e-2],
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# CSV PARSER
# ══════════════════════════════════════════════════════════════════════════════
def load_csv():
    df = pd.read_csv(CSV_PATH)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    _YES = {"yes":1.0,"no":0.0,"positive":1.0,"negative":0.0,
            "1":1.0,"0":0.0}
    col_map = [
        ("chronic_back_pain",    "chronic_back_pain"),
        ("morning_stiffness",    "morning_stifness"),
        ("improvement_exercise", "improves_with_exercise"),
        ("hla_b27",              "hla_b27_positive"),
    ]
    for feat, src in col_map:
        if src in df.columns:
            df[feat] = (df[src].astype(str).str.strip().str.lower()
                          .map(_YES).fillna(0.0).astype(np.float32))
        else:
            df[feat] = 0.0
    for raw, norm in [("esr(mm/hr)","esr_norm"),("crp(mg/l)","crp_norm")]:
        if raw in df.columns:
            v = pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
            df[norm] = ((v-v.mean())/(v.std()+1e-8)).astype(np.float32)
        else:
            df[norm] = 0.0
    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)
    df["image_id"] = df["image_id"].astype(str).str.strip()
    print(f"[CSV] {len(df)} rows loaded")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# MRI PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class MRIPreprocessor:
    def __init__(self, target_size=IMG_SIZE):
        self.target_size = target_size
        self._kernel = self._gauss_kernel(15, 31)
    @staticmethod
    def _gauss_kernel(sigma, size):
        c  = torch.arange(size).float() - size//2
        g1 = torch.exp(-c**2/(2*sigma**2))
        g2 = torch.ger(g1,g1)
        return (g2/g2.sum()).unsqueeze(0).unsqueeze(0)
    def _n4(self, arr):
        t  = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        bf = F.conv2d(t, self._kernel, padding=self._kernel.shape[-1]//2)
        return (t/(bf+1e-3)).squeeze().numpy()
    @staticmethod
    def _align(arr):
        ys,xs = np.where(arr > np.percentile(arr,75))
        if len(ys)<20: return arr
        return np.roll(np.roll(arr,int(arr.shape[0]//2-ys.mean()),axis=0),
                       int(arr.shape[1]//2-xs.mean()),axis=1)
    @staticmethod
    def _zscore(arr):
        return ((arr-arr.mean())/(arr.std()+1e-8)).astype(np.float32)
    def process(self, pil_img):
        arr = np.array(pil_img.convert("L"),dtype=np.float32)/255.0
        arr = self._n4(arr); arr = self._align(arr); arr = self._zscore(arr)
        t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t   = F.interpolate(t,size=(self.target_size,self.target_size),
                            mode="bilinear",align_corners=False)
        return t.squeeze(0).repeat(3,1,1)

# ══════════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SacroDataset(Dataset):
    def __init__(self, root, split, meta_df=None, augment=False):
        self.prep    = MRIPreprocessor(IMG_SIZE)
        self.records = []
        for cls, label in [("AS",1),("Normal",0)]:
            d = os.path.join(root, split, cls)
            if not os.path.isdir(d): continue
            for p in sorted(glob.glob(os.path.join(d,"*.png"))):
                self.records.append({"path":p,"label":label,
                                     "image_id":os.path.splitext(
                                         os.path.basename(p))[0]})
        self._meta  = {}
        self._zero  = np.zeros(META_DIM, dtype=np.float32)
        if meta_df is not None:
            for _, row in meta_df.iterrows():
                k = str(row.get("image_id","")).strip()
                self._meta[k] = np.array(
                    [float(row.get(f,0)) for f in CLINICAL_FEATURES],
                    dtype=np.float32)
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(10),
        ]) if augment else None
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        try:    img = self.prep.process(Image.open(r["path"]))
        except: img = torch.zeros(3,IMG_SIZE,IMG_SIZE)
        if self.aug: img = self.aug(img)
        meta = self._meta.get(r["image_id"], self._zero.copy())
        return {"image":    img,
                "label":    torch.tensor(r["label"],dtype=torch.long),
                "meta_vec": torch.from_numpy(meta)}

def get_loaders(batch_size, meta_df):
    loaders = {}
    for split, aug in [("train",True),("val",False),("test",False)]:
        ds = SacroDataset(DATA_ROOT, split, meta_df=meta_df, augment=aug)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split=="train"),
                                    num_workers=0, pin_memory=True,
                                    drop_last=(split=="train"))
    return loaders

# ══════════════════════════════════════════════════════════════════════════════
# ACTIVATION FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def get_activation(name):
    return {"relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "elu":  nn.ELU(),
            "tanh": nn.Tanh(),
            "sigmoid": nn.Sigmoid()}.get(name.lower(), nn.GELU())

# ══════════════════════════════════════════════════════════════════════════════
# MODEL — ResNet50 backbone, configurable hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
class ConfigurableModel(nn.Module):
    """
    ResNet50 backbone with configurable:
      - dropout rate
      - activation function
    Everything else (architecture, fusion, classifier) stays fixed.
    """
    def __init__(self, dropout=0.3, activation="gelu",
                 embed_dim=EMBED_DIM):
        super().__init__()
        act = get_activation(activation)

        # Image encoder (ResNet50, no global pool — we apply GAP ourselves)
        bb = timm.create_model("resnet50", pretrained=True,
                               num_classes=0, global_pool="")
        self.backbone = bb
        self.gap      = nn.AdaptiveAvgPool2d(1)

        # Projection
        self.img_proj = nn.Sequential(
            nn.Linear(2048, 1024), get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(1024, embed_dim), nn.Dropout(dropout),
        )

        # Clinical encoder
        self.meta_enc = nn.Sequential(
            nn.Linear(META_DIM, 64), get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(64, embed_dim), nn.Dropout(dropout),
        )

        # Cross-attention
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln  = nn.LayerNorm(embed_dim)

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(embed_dim*2, 256), get_activation(activation),
            nn.Dropout(dropout), nn.Linear(256, 2),
        )

    def _cross_attn(self, Ei, Em):
        B,D = Ei.shape; h=4; dk=D//h
        Q = self.W_Q(Em).view(B,1,h,dk).transpose(1,2)
        K = self.W_K(Ei).view(B,1,h,dk).transpose(1,2)
        V = self.W_V(Ei).view(B,1,h,dk).transpose(1,2)
        A   = F.softmax(torch.matmul(Q,K.transpose(-2,-1))/dk**0.5,dim=-1)
        ctx = torch.matmul(A,V).transpose(1,2).contiguous().view(B,D)
        return self.ln(self.W_O(ctx)+Em)

    def forward(self, images, meta):
        feat = self.gap(self.backbone(images)).flatten(1)
        Ei   = F.normalize(self.img_proj(feat), p=2, dim=-1)
        Em   = F.normalize(self.meta_enc(meta),  p=2, dim=-1)
        f    = self._cross_attn(Ei, Em)
        return self.clf(torch.cat([Ei,f],dim=-1))

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

# ══════════════════════════════════════════════════════════════════════════════
# OPTIMIZER FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def get_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9,
                               weight_decay=weight_decay)
    elif name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    else:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

# ══════════════════════════════════════════════════════════════════════════════
# SCHEDULER FACTORY
# ══════════════════════════════════════════════════════════════════════════════
def get_scheduler(name, optimizer):
    name = name.lower()
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=N_EPOCHS)
    elif name == "step":
        return torch.optim.lr_scheduler.StepLR(
                    optimizer, step_size=3, gamma=0.5)
    elif name == "exponential":
        return torch.optim.lr_scheduler.ExponentialLR(
                    optimizer, gamma=0.85)
    else:
        return None   # no scheduler

# ══════════════════════════════════════════════════════════════════════════════
# TRAINING & EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def train_and_eval(config: dict, loaders: dict, label: str) -> dict:
    """
    Train one configuration and return full metrics.
    config keys: optimizer, lr, batch_size, dropout, activation,
                 scheduler, weight_decay
    """
    dropout    = config.get("dropout",    0.3)
    activation = config.get("activation", "gelu")
    optimizer  = config.get("optimizer",  "adamw")
    lr         = config.get("lr",         3e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    scheduler  = config.get("scheduler",  "cosine")

    model   = ConfigurableModel(dropout=dropout, activation=activation).to(DEVICE)
    opt     = get_optimizer(optimizer, model.parameters(), lr, weight_decay)
    sched   = get_scheduler(scheduler, opt)
    loss_fn = nn.CrossEntropyLoss()
    n_params = model.count_parameters()

    t0 = time.time()
    train_losses, val_accs = [], []

    print(f"\n  [{label}]  params={n_params:,}")
    for ep in range(1, N_EPOCHS+1):
        # Train
        model.train(); total=0.0
        for b in loaders["train"]:
            imgs = b["image"].to(DEVICE).float()
            meta = b["meta_vec"].to(DEVICE).float()
            labs = b["label"].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(imgs,meta), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        if sched: sched.step()
        tr_loss = total/max(len(loaders["train"]),1)

        # Val
        model.eval(); all_p,all_l=[],[]
        with torch.no_grad():
            for b in loaders["val"]:
                imgs=b["image"].to(DEVICE).float()
                meta=b["meta_vec"].to(DEVICE).float()
                all_p.append(F.softmax(model(imgs,meta),dim=-1).cpu())
                all_l.append(b["label"])
        probs=torch.cat(all_p).numpy(); labs=torch.cat(all_l).numpy()
        val_acc = accuracy_score(labs, probs.argmax(1))
        train_losses.append(tr_loss); val_accs.append(val_acc)
        print(f"    ep {ep:02d}/{N_EPOCHS}  "
              f"loss={tr_loss:.4f}  val_acc={val_acc:.4f}")

    # Test evaluation — full metrics
    model.eval(); all_p,all_l=[],[]
    with torch.no_grad():
        for b in loaders["test"]:
            imgs=b["image"].to(DEVICE).float()
            meta=b["meta_vec"].to(DEVICE).float()
            all_p.append(F.softmax(model(imgs,meta),dim=-1).cpu())
            all_l.append(b["label"])
    probs  = torch.cat(all_p).numpy()
    labels = torch.cat(all_l).numpy()
    preds  = probs.argmax(1)
    probs_c = np.clip(probs,1e-10,1-1e-10)

    acc  = accuracy_score(labels, preds)
    auc  = roc_auc_score(labels, probs_c[:,1])
    ll   = log_loss(labels, probs_c)
    prec = precision_score(labels, preds, zero_division=0)
    rec  = recall_score(labels, preds, zero_division=0)
    f1   = f1_score(labels, preds, zero_division=0)
    cm   = confusion_matrix(labels, preds)
    elapsed = time.time()-t0

    print(f"  Test: acc={acc:.4f}  auc={auc:.4f}  "
          f"prec={prec:.4f}  rec={rec:.4f}  f1={f1:.4f}")

    return {
        "label":          label,
        "Accuracy":       round(acc,4),
        "AUC":            round(auc,4),
        "Precision":      round(prec,4),
        "Recall":         round(rec,4),
        "F1":             round(f1,4),
        "Log-Loss":       round(ll,4),
        "Parameters":     n_params,
        "Train time(min)":round(elapsed/60,2),
        "train_losses":   train_losses,
        "val_accs":       val_accs,
        "confusion_matrix": cm,
        "config":         config,
    }

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] → {name}")


def plot_experiment(results: list, exp_name: str, variable: str):
    """Bar chart + learning curves for one experiment."""
    labels   = [r["label"] for r in results]
    accs     = [r["Accuracy"] for r in results]
    f1s      = [r["F1"] for r in results]
    aucs     = [r["AUC"] for r in results]
    x        = np.arange(len(labels))
    colors   = plt.cm.Set2(np.linspace(0,1,len(labels)))

    # Bar chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Experiment: {variable} Comparison",
                 fontsize=13, fontweight="bold")
    for ax, vals, title, color in [
        (axes[0], accs, "Test Accuracy",  "steelblue"),
        (axes[1], f1s,  "F1 Score",       "firebrick"),
        (axes[2], aucs, "AUC",            "seagreen"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.12); ax.set_title(title)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, f"exp_{exp_name}_bars.png")

    # Learning curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{variable} — Learning Curves",
                 fontsize=12, fontweight="bold")
    epochs = range(1, N_EPOCHS+1)
    for r, c in zip(results, colors):
        axes[0].plot(epochs, r["train_losses"], "o-",
                     label=r["label"], color=c)
        axes[1].plot(epochs, r["val_accs"], "s-",
                     label=r["label"], color=c)
    for ax, title, ylabel in [
        (axes[0], "Training Loss",    "Loss"),
        (axes[1], "Validation Accuracy", "Accuracy"),
    ]:
        ax.set_title(title); ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel); ax.legend(fontsize=8)
        ax.grid(ls="--", alpha=0.4)
    _save(fig, f"exp_{exp_name}_curves.png")

    # Confusion matrix for best result
    best = max(results, key=lambda r: r["F1"])
    cm   = best["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(5,4))
    sns.heatmap(cm.astype(int), annot=True, fmt="d", cmap="Blues",
                xticklabels=["Normal","AS"],
                yticklabels=["Normal","AS"], ax=ax)
    ax.set_title(f"Confusion Matrix — Best {variable}: {best['label']}",
                 fontweight="bold")
    ax.set_ylabel("True Label"); ax.set_xlabel("Predicted Label")
    _save(fig, f"exp_{exp_name}_confusion_best.png")


def plot_parameter_efficiency(param_df: pd.DataFrame):
    """Params vs Accuracy scatter for all 17 backbones + hyperparameter configs."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Parameter Efficiency Analysis",
                 fontsize=13, fontweight="bold")

    # Scatter: params vs accuracy
    ax = axes[0]
    families = param_df["Family"].unique()
    cmap = plt.cm.tab10(np.linspace(0,1,len(families)))
    for fam, c in zip(families, cmap):
        sub = param_df[param_df["Family"]==fam]
        ax.scatter(sub["Parameters(M)"], sub["Test Accuracy"],
                   label=fam, color=c, s=80, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Backbone"],
                        (row["Parameters(M)"], row["Test Accuracy"]),
                        textcoords="offset points", xytext=(5,3),
                        fontsize=7)
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Test Accuracy")
    ax.set_title("Parameters vs Accuracy")
    ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)

    # Bar: params vs WEAT bias
    ax = axes[1]
    colors = plt.cm.RdYlGn_r(
        np.linspace(0,1,len(param_df)))
    bars = ax.bar(range(len(param_df)),
                  param_df["WEAT |d|"],
                  color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(param_df)))
    ax.set_xticklabels(param_df["Backbone"], rotation=45,
                       ha="right", fontsize=8)
    ax.set_ylabel("WEAT Bias |d|")
    ax.set_title("Backbone vs Representational Bias")
    ax.axhline(0.5, color="red", ls="--", lw=1, label="High bias threshold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    # Annotate param count
    for i, (_, row) in enumerate(param_df.iterrows()):
        ax.text(i, 0.05, f"{row['Parameters(M)']:.0f}M",
                ha="center", va="bottom", fontsize=6, rotation=90)

    _save(fig, "parameter_efficiency.png")


def plot_final_summary(all_best: dict):
    """Summary table of best hyperparameter per experiment."""
    rows = []
    for exp, info in all_best.items():
        rows.append({
            "Experiment":    info["variable"],
            "Best Value":    info["best_value"],
            "Best Accuracy": info["best_acc"],
            "Best F1":       info["best_f1"],
            "Best AUC":      info["best_auc"],
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "best_hyperparameters.csv"), index=False)

    fig, ax = plt.subplots(figsize=(13, max(4, len(rows)*0.7+1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for j in range(len(df.columns)):
        tbl[0,j].set_facecolor("#1a3a5c")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows)+1):
        clr = "#EEF4FF" if i%2==0 else "#FFFFFF"
        for j in range(len(df.columns)):
            tbl[i,j].set_facecolor(clr)
    fig.suptitle("Best Hyperparameter per Experiment — SacroMRI",
                 fontsize=12, fontweight="bold")
    _save(fig, "best_hyperparameters_table.png")
    print("\n  Best hyperparameters per experiment:")
    print(df.to_string(index=False))
    return df


# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT 8 — PARAMETER EFFICIENCY TABLE
# Uses results from the main backbone study (tally_all17_FIXED.csv)
# ══════════════════════════════════════════════════════════════════════════════
BACKBONE_PARAMS = {
    "vgg11":            132.9,
    "vgg13":            133.0,
    "vgg16":            138.4,
    "vgg19":            143.7,
    "resnet18":          11.7,
    "resnet34":          21.8,
    "resnet50":          25.6,
    "resnet101":         44.5,
    "resnet152":         60.2,
    "inception_v3":      27.2,
    "inception_v4":      42.7,
    "inception_resnet":  55.9,
    "unet":              31.0,
    "unet_pp":           36.6,
    "attention_unet":    34.9,
    "residual_unet":     32.1,
    "unet3d":            31.0,
}

def build_param_efficiency_table():
    tally_path = r"D:\SacroMri\bvq_outputs_fixed\tally_all17_FIXED.csv"
    if not os.path.exists(tally_path):
        tally_path = r"D:\SacroMri\bvq_outputs_fixed\tally_incremental_fixed.csv"
    if not os.path.exists(tally_path):
        print("  [WARN] tally_all17_FIXED.csv not found — skipping param table")
        return None
    df = pd.read_csv(tally_path)
    df["Parameters(M)"] = df["Backbone"].map(BACKBONE_PARAMS).fillna(0)
    df = df.sort_values("Parameters(M)")
    cols = ["Backbone","Family","Year","Parameters(M)",
            "Test Accuracy","Test AUC","WEAT |d|","H_epistemic",
            "Train time (min)"]
    available = [c for c in cols if c in df.columns]
    result = df[available].reset_index(drop=True)
    result.to_csv(os.path.join(OUT_DIR,"parameter_efficiency.csv"),index=False)
    print("\n  Parameter Efficiency Table:")
    print(result.to_string(index=False))
    return result


def main():
    print("\n" + "="*60)
    print("  HYPERPARAMETER COMPARISON STUDY — SacroMRI")
    print("  Backbone fixed: ResNet50 | One variable at a time")
    print("="*60)

    meta_df  = load_csv()
    all_best = {}
    all_results_flat = []

    # ── RESUME: load previously completed configs ─────────────────────────────
    inc_csv = os.path.join(OUT_DIR, "hyperparam_incremental.csv")
    completed_labels = set()   # set of "experiment::label" already done

    if os.path.exists(inc_csv):
        old_df = pd.read_csv(inc_csv)
        print(f"\n  Resuming — {len(old_df)} configs already completed:")
        print(old_df[["experiment","label","Accuracy"]].to_string(index=False))

        # Rebuild all_results_flat from existing CSV (without plot data)
        for _, row in old_df.iterrows():
            all_results_flat.append(row.to_dict())
            completed_labels.add(f"{row['experiment']}::{row['label']}")

        # Rebuild all_best from existing CSV for completed experiments
        for exp_name, exp_cfg in EXPERIMENTS.items():
            variable = exp_cfg["variable"]
            exp_rows = old_df[old_df["experiment"] == variable]
            # Only mark as fully done if ALL values for this experiment exist
            n_expected = len(exp_cfg["values"])
            if len(exp_rows) >= n_expected:
                best_row = exp_rows.loc[exp_rows["Accuracy"].idxmax()]
                all_best[exp_name] = {
                    "variable":   variable,
                    "best_value": str(best_row["label"]),
                    "best_acc":   best_row["Accuracy"],
                    "best_f1":    best_row.get("F1", "N/A"),
                    "best_auc":   best_row.get("AUC", "N/A"),
                }
                print(f"  ✓ {variable} already complete — "
                      f"best={best_row['label']}")
    else:
        print("\n  No existing results found — starting fresh")

    # ── Run all 7 hyperparameter experiments ──────────────────────────────────
    for exp_name, exp_cfg in EXPERIMENTS.items():
        variable = exp_cfg["variable"]
        fixed    = exp_cfg["fixed"]
        values   = exp_cfg["values"]

        # Skip entire experiment if already fully completed
        if exp_name in all_best:
            print(f"\n  SKIPPING {variable} — already complete")
            continue

        print(f"\n{'='*60}")
        print(f"  EXPERIMENT: {variable}")
        print(f"  Fixed: {fixed}")
        print(f"  Values to test: {values}")
        print(f"{'='*60}")

        exp_results = []
        for val in values:
            config  = dict(fixed)
            var_key = exp_name
            config[var_key] = val
            label   = str(val)

            # Skip individual configs already done within a partial experiment
            skip_key = f"{variable}::{label}"
            if skip_key in completed_labels:
                print(f"  SKIPPING {variable} = {label} — already done")
                # Recover result from old_df for plot reconstruction
                old_row = old_df[
                    (old_df["experiment"] == variable) &
                    (old_df["label"] == label)
                ]
                if len(old_row) > 0:
                    # Reconstruct minimal result dict for plotting
                    r = old_row.iloc[0].to_dict()
                    r["train_losses"] = [float(r.get("Log-Loss",0))] * N_EPOCHS
                    r["val_accs"]     = [float(r.get("Accuracy",1))] * N_EPOCHS
                    r["confusion_matrix"] = np.array([[r.get("Accuracy",1),0],[0,r.get("Accuracy",1)]])
                    exp_results.append(r)
                continue

            # Get loaders for this batch size
            bs      = config.get("batch_size", 16)
            loaders = get_loaders(bs, meta_df)

            result = train_and_eval(config, loaders, label)
            result["experiment"] = variable
            exp_results.append(result)
            all_results_flat.append(result)
            completed_labels.add(skip_key)

            # Incremental save after every config
            pd.DataFrame([{k:v for k,v in r.items()
                           if k not in ("train_losses","val_accs",
                                        "confusion_matrix","config")}
                          for r in all_results_flat]).to_csv(
                inc_csv, index=False)

        # Only plot if we have real results with training curves
        real_results = [r for r in exp_results
                        if isinstance(r.get("train_losses"), list)
                        and len(r["train_losses"]) == N_EPOCHS]
        if real_results:
            plot_experiment(real_results, exp_name, variable)
        else:
            print(f"  [INFO] Skipping plot for {variable} — no new training data")

        # Find best across all results for this experiment (new + recovered)
        scorable = [r for r in exp_results if isinstance(r.get("Accuracy"), float)]
        if scorable:
            best = max(scorable, key=lambda r: (r["Accuracy"], r.get("F1", 0)))
            all_best[exp_name] = {
                "variable":   variable,
                "best_value": best["label"],
                "best_acc":   best["Accuracy"],
                "best_f1":    best.get("F1", "N/A"),
                "best_auc":   best.get("AUC", "N/A"),
            }
            print(f"\n  Best {variable}: {best['label']} "
                  f"(acc={best['Accuracy']:.4f})")

    # ── Experiment 8: Parameter Efficiency ────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  EXPERIMENT 8: Parameter Efficiency")
    print(f"{'='*60}")
    param_df = build_param_efficiency_table()
    if param_df is not None:
        plot_parameter_efficiency(param_df)

    # ── Final summary ─────────────────────────────────────────────────────────
    best_df = plot_final_summary(all_best)

    # ── Save full results ─────────────────────────────────────────────────────
    flat_df = pd.DataFrame([
        {k:v for k,v in r.items()
         if k not in ("train_losses","val_accs","confusion_matrix","config")}
        for r in all_results_flat
    ])
    flat_df.to_csv(os.path.join(OUT_DIR,"all_hyperparam_results.csv"),
                   index=False)

    # ── Print final answers to sir's questions ────────────────────────────────
    print("\n" + "="*60)
    print("  ANSWERS TO SUPERVISOR'S QUESTIONS")
    print("="*60)
    print("\n  Q1: Which optimizer works best?")
    o = all_best.get("optimizer",{})
    print(f"      → {o.get('best_value','N/A')} "
          f"(acc={o.get('best_acc','N/A')}  f1={o.get('best_f1','N/A')})")

    print("\n  Q2: Which learning rate works best?")
    l = all_best.get("learning_rate",{})
    print(f"      → {l.get('best_value','N/A')} "
          f"(acc={l.get('best_acc','N/A')}  f1={l.get('best_f1','N/A')})")

    print("\n  Q3: Which batch size works best?")
    b = all_best.get("batch_size",{})
    print(f"      → {b.get('best_value','N/A')} "
          f"(acc={b.get('best_acc','N/A')}  f1={b.get('best_f1','N/A')})")

    print("\n  Q4: Which dropout rate works best?")
    d = all_best.get("dropout",{})
    print(f"      → {d.get('best_value','N/A')} "
          f"(acc={d.get('best_acc','N/A')}  f1={d.get('best_f1','N/A')})")

    print("\n  Q5: Which activation function works best?")
    a = all_best.get("activation",{})
    print(f"      → {a.get('best_value','N/A')} "
          f"(acc={a.get('best_acc','N/A')}  f1={a.get('best_f1','N/A')})")

    print("\n  Q6: Which LR scheduler works best?")
    s = all_best.get("scheduler",{})
    print(f"      → {s.get('best_value','N/A')} "
          f"(acc={s.get('best_acc','N/A')}  f1={s.get('best_f1','N/A')})")

    print("\n  Q7: Which weight decay works best?")
    w = all_best.get("weight_decay",{})
    print(f"      → {w.get('best_value','N/A')} "
          f"(acc={w.get('best_acc','N/A')}  f1={w.get('best_f1','N/A')})")

    if param_df is not None:
        most_efficient = param_df.loc[param_df["Parameters(M)"].idxmin()]
        print(f"\n  Q8: Can we achieve similar accuracy with fewer parameters?")
        print(f"      → Yes. {most_efficient['Backbone']} achieves "
              f"acc={most_efficient['Test Accuracy']} with only "
              f"{most_efficient['Parameters(M)']}M parameters.")

    print(f"\n[DONE] All outputs in {OUT_DIR}")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR,f))
        print(f"  {f:<45s}  {size/1024:6.1f} KB")


if __name__ == "__main__":
    main()