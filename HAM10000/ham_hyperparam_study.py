"""
=============================================================================
 HYPERPARAMETER COMPARISON STUDY — HAM10000
 (companion to ham_pipeline.py; UNIFIED settings so the ResNet50 baseline
  matches the main 17-backbone study: same backbone, 8 epochs, same default
  hyperparameters, same RGB preprocessing, same patient-level 7-class split)

 EXPERIMENT 1 : Optimizer        (AdamW, SGD, RMSprop, Adam)
 EXPERIMENT 2 : Learning rate    (1e-3, 3e-4, 1e-5)            [trimmed]
 EXPERIMENT 3 : Batch size       (8, 16, 32, 64)
 EXPERIMENT 4 : Dropout rate     (0.0, 0.3, 0.5)               [trimmed]
 EXPERIMENT 5 : Activation       (ReLU, GELU, ELU, Tanh)
 EXPERIMENT 6 : LR Scheduler     (Cosine, StepLR, ExponentialLR, None)
 EXPERIMENT 7 : Weight decay     (0, 1e-4, 1e-2)               [trimmed]
 EXPERIMENT 8 : Parameter efficiency (reads HAM_tally_incremental.csv)

 Backbone fixed to ResNet50; one variable changes at a time.
 Default config == main-study ResNet50:
   optimizer=adamw, lr=3e-4, batch=16, dropout=0.3, activation=gelu,
   scheduler=cosine, weight_decay=1e-4, 8 epochs.

 Metrics reported are imbalance-aware (HAM10000 is dominated by 'nv'):
   Accuracy, Macro-F1, Balanced Accuracy, macro-AUC (ovr), Log-Loss.
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, roc_auc_score, log_loss,
                             precision_score, recall_score, f1_score,
                             balanced_accuracy_score, confusion_matrix)
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
print("\u2713  HAM10000 hyperparameter comparison script loaded")

# ==============================================================================
# PATHS  — match ham_pipeline.py
# ==============================================================================
HAM_ROOT = r"D:\SacroMri\HAM1000"
CSV_PATH = os.path.join(HAM_ROOT, "HAM10000_metadata.csv")
IMG_DIRS = [os.path.join(HAM_ROOT, "HAM10000_images_part_1"),
            os.path.join(HAM_ROOT, "HAM10000_images_part_2")]
OUT_DIR  = r"D:\SacroMri\HAM_hyperparam_study"
os.makedirs(OUT_DIR, exist_ok=True)

# main-study tally, for Experiment 8 (parameter efficiency)
MAIN_TALLY = r"D:\SacroMri\HAM_bvq_outputs\HAM_tally_incremental.csv"

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE  = 340          # ResNet50 uses 340 (matches main study)
EMBED_DIM = 512
N_EPOCHS  = 8            # UNIFIED with main study
SEED      = 42
N_CLASSES = 7
torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"\u2713  device={DEVICE}  epochs={N_EPOCHS}  (unified with main study)")

DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_TO_IDX  = {c: i for i, c in enumerate(DX_CLASSES)}
LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "unknown", "upper extremity",
]
LOC_TO_IDX = {c: i for i, c in enumerate(LOCALIZATIONS)}
META_DIM = 2 + len(LOCALIZATIONS)

# ==============================================================================
# HYPERPARAMETER GRIDS (trimmed where redundant; categoricals kept full)
# ==============================================================================
EXPERIMENTS = {
    "optimizer": {
        "variable": "Optimizer",
        "fixed": {"lr": 3e-4, "batch_size": 16, "dropout": 0.3,
                  "activation": "gelu", "scheduler": "cosine", "weight_decay": 1e-4},
        "values": ["adamw", "sgd", "rmsprop", "adam"],
    },
    "learning_rate": {
        "variable": "Learning Rate",
        "fixed": {"optimizer": "adamw", "batch_size": 16, "dropout": 0.3,
                  "activation": "gelu", "scheduler": "cosine", "weight_decay": 1e-4},
        "values": [1e-3, 3e-4, 1e-5],
    },
    "batch_size": {
        "variable": "Batch Size",
        "fixed": {"optimizer": "adamw", "lr": 3e-4, "dropout": 0.3,
                  "activation": "gelu", "scheduler": "cosine", "weight_decay": 1e-4},
        "values": [8, 16, 32, 64],
    },
    "dropout": {
        "variable": "Dropout Rate",
        "fixed": {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                  "activation": "gelu", "scheduler": "cosine", "weight_decay": 1e-4},
        "values": [0.0, 0.3, 0.5],
    },
    "activation": {
        "variable": "Activation Function",
        "fixed": {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                  "dropout": 0.3, "scheduler": "cosine", "weight_decay": 1e-4},
        "values": ["relu", "gelu", "elu", "tanh"],
    },
    "scheduler": {
        "variable": "LR Scheduler",
        "fixed": {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                  "dropout": 0.3, "activation": "gelu", "weight_decay": 1e-4},
        "values": ["cosine", "step", "exponential", "none"],
    },
    "weight_decay": {
        "variable": "Weight Decay (\u03bb)",
        "fixed": {"optimizer": "adamw", "lr": 3e-4, "batch_size": 16,
                  "dropout": 0.3, "activation": "gelu", "scheduler": "cosine"},
        "values": [0.0, 1e-4, 1e-2],
    },
}

# ==============================================================================
# DATA  (identical logic to ham_pipeline.py)
# ==============================================================================
def build_image_index():
    idx = {}
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.jpg")):
            idx[os.path.splitext(os.path.basename(p))[0]] = p
    print(f"[IMG] indexed {len(idx)} jpgs")
    return idx


def encode_meta_row(row, age_mean, age_std):
    vec = np.zeros(META_DIM, dtype=np.float32)
    age = row.get("age", np.nan)
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = np.nan
    vec[0] = 0.0 if (age != age) else (age - age_mean) / (age_std + 1e-8)
    sex = str(row.get("sex", "")).strip().lower()
    vec[1] = 1.0 if sex == "female" else (0.0 if sex == "male" else 0.5)
    loc = str(row.get("localization", "")).strip().lower()
    vec[2 + LOC_TO_IDX.get(loc, LOC_TO_IDX["unknown"])] = 1.0
    return vec


def load_and_split_csv(csv_path, img_index):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["path"] = df["image_id"].map(img_index)
    df = df[df["path"].notna()].reset_index(drop=True)
    df["label"] = df["dx"].astype(str).str.strip().str.lower().map(DX_TO_IDX)
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    age_num = pd.to_numeric(df["age"], errors="coerce")
    age_mean, age_std = float(age_num.mean()), float(age_num.std())
    df["meta_vec"] = df.apply(lambda r: encode_meta_row(r, age_mean, age_std), axis=1)

    lesions = (df.groupby("lesion_id")["label"]
                 .agg(lambda s: s.value_counts().index[0]).reset_index())
    les_train, les_tmp = train_test_split(
        lesions, test_size=0.20, random_state=SEED, stratify=lesions["label"])
    les_val, les_test = train_test_split(
        les_tmp, test_size=0.50, random_state=SEED, stratify=les_tmp["label"])
    sm = {}
    for lid in les_train["lesion_id"]: sm[lid] = "train"
    for lid in les_val["lesion_id"]:   sm[lid] = "val"
    for lid in les_test["lesion_id"]:  sm[lid] = "test"
    df["split"] = df["lesion_id"].map(sm)
    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        print(f"  [{s:5s}] {len(sub):5d}  counts={np.bincount(sub['label'], minlength=N_CLASSES).tolist()}")
    return df


class DermPreprocessor:
    def __init__(self, target_size=IMG_SIZE): self.target_size = target_size
    @staticmethod
    def _zscore(a):
        mu, sd = a.mean(), a.std()
        return ((a - mu) / (sd + 1e-8)).astype(np.float32)
    def process(self, pil_img):
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        for c in range(3):
            arr[c] = self._zscore(arr[c])
        t = torch.from_numpy(arr).unsqueeze(0)
        t = F.interpolate(t, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return t.squeeze(0)


class HAMDataset(Dataset):
    def __init__(self, split_df, split, augment=False):
        self.prep = DermPreprocessor(IMG_SIZE)
        sub = split_df[split_df["split"] == split].reset_index(drop=True)
        self.records = [{
            "path": r["path"], "label": int(r["label"]),
            "meta": r["meta_vec"].astype(np.float32),
        } for _, r in sub.iterrows()]
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomVerticalFlip(0.2),
            transforms.RandomRotation(10),
        ]) if augment else None
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        r = self.records[idx]
        try:
            img = self.prep.process(Image.open(r["path"]))
        except Exception:
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE)
        if self.aug: img = self.aug(img)
        return {"image": img,
                "label": torch.tensor(r["label"], dtype=torch.long),
                "meta_vec": torch.from_numpy(r["meta"])}


def get_loaders(batch_size, split_df):
    loaders = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        ds = HAMDataset(split_df, split, augment=aug)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=0, pin_memory=True,
                                    drop_last=(split == "train"))
    return loaders


# ==============================================================================
# MODEL  (ResNet50, configurable dropout + activation; 7-class head)
# ==============================================================================
def get_activation(name):
    return {"relu": nn.ReLU(), "gelu": nn.GELU(), "elu": nn.ELU(),
            "tanh": nn.Tanh(), "sigmoid": nn.Sigmoid()}.get(name.lower(), nn.GELU())


class ConfigurableModel(nn.Module):
    def __init__(self, dropout=0.3, activation="gelu", embed_dim=EMBED_DIM):
        super().__init__()
        bb = timm.create_model("resnet50", pretrained=True,
                               num_classes=0, global_pool="")
        self.backbone = bb
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.img_proj = nn.Sequential(
            nn.Linear(2048, 1024), get_activation(activation), nn.Dropout(dropout),
            nn.Linear(1024, embed_dim), nn.Dropout(dropout))
        self.meta_enc = nn.Sequential(
            nn.Linear(META_DIM, 64), get_activation(activation), nn.Dropout(dropout),
            nn.Linear(64, embed_dim), nn.Dropout(dropout))
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln  = nn.LayerNorm(embed_dim)
        self.clf = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), get_activation(activation),
            nn.Dropout(dropout), nn.Linear(256, N_CLASSES))   # 7-class head
    def _cross_attn(self, Ei, Em):
        B, D = Ei.shape; h = 4; dk = D // h
        Q = self.W_Q(Em).view(B, 1, h, dk).transpose(1, 2)
        K = self.W_K(Ei).view(B, 1, h, dk).transpose(1, 2)
        V = self.W_V(Ei).view(B, 1, h, dk).transpose(1, 2)
        A = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / dk ** 0.5, dim=-1)
        ctx = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, D)
        return self.ln(self.W_O(ctx) + Em)
    def forward(self, images, meta):
        feat = self.gap(self.backbone(images)).flatten(1)
        Ei = F.normalize(self.img_proj(feat), p=2, dim=-1)
        Em = F.normalize(self.meta_enc(meta), p=2, dim=-1)
        f = self._cross_attn(Ei, Em)
        return self.clf(torch.cat([Ei, f], dim=-1))
    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def get_optimizer(name, params, lr, weight_decay):
    name = name.lower()
    if name == "sgd":     return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "rmsprop": return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "adam":    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)


def get_scheduler(name, optimizer):
    name = name.lower()
    if name == "cosine":      return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
    if name == "step":        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    if name == "exponential": return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.85)
    return None


# ==============================================================================
# TRAIN + EVAL  (imbalance-aware metrics)
# ==============================================================================
def train_and_eval(config, loaders, label):
    dropout    = config.get("dropout", 0.3)
    activation = config.get("activation", "gelu")
    optimizer  = config.get("optimizer", "adamw")
    lr         = config.get("lr", 3e-4)
    weight_decay = config.get("weight_decay", 1e-4)
    scheduler  = config.get("scheduler", "cosine")

    model   = ConfigurableModel(dropout=dropout, activation=activation).to(DEVICE)
    opt     = get_optimizer(optimizer, model.parameters(), lr, weight_decay)
    sched   = get_scheduler(scheduler, opt)
    loss_fn = nn.CrossEntropyLoss()
    n_params = model.count_parameters()

    t0 = time.time()
    train_losses, val_accs = [], []
    print(f"\n  [{label}]  params={n_params:,}")
    for ep in range(1, N_EPOCHS + 1):
        model.train(); total = 0.0
        for b in loaders["train"]:
            imgs = b["image"].to(DEVICE).float()
            meta = b["meta_vec"].to(DEVICE).float()
            labs = b["label"].to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(imgs, meta), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()
        if sched: sched.step()
        tr_loss = total / max(len(loaders["train"]), 1)

        model.eval(); all_p, all_l = [], []
        with torch.no_grad():
            for b in loaders["val"]:
                imgs = b["image"].to(DEVICE).float()
                meta = b["meta_vec"].to(DEVICE).float()
                all_p.append(F.softmax(model(imgs, meta), dim=-1).cpu())
                all_l.append(b["label"])
        probs = torch.cat(all_p).numpy(); labs = torch.cat(all_l).numpy()
        val_acc = accuracy_score(labs, probs.argmax(1))
        train_losses.append(tr_loss); val_accs.append(val_acc)
        print(f"    ep {ep:02d}/{N_EPOCHS}  loss={tr_loss:.4f}  val_acc={val_acc:.4f}")

    # test
    model.eval(); all_p, all_l = [], []
    with torch.no_grad():
        for b in loaders["test"]:
            imgs = b["image"].to(DEVICE).float()
            meta = b["meta_vec"].to(DEVICE).float()
            all_p.append(F.softmax(model(imgs, meta), dim=-1).cpu())
            all_l.append(b["label"])
    probs  = torch.cat(all_p).numpy(); labels = torch.cat(all_l).numpy()
    preds  = probs.argmax(1)
    probs_c = np.clip(probs, 1e-10, 1 - 1e-10)
    probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)

    acc      = accuracy_score(labels, preds)
    bal_acc  = balanced_accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro",
                        labels=list(range(N_CLASSES)), zero_division=0)
    try:
        auc = roc_auc_score(labels, probs_c, multi_class="ovr",
                            average="macro", labels=list(range(N_CLASSES)))
    except Exception:
        auc = 0.5
    ll = log_loss(labels, probs_c, labels=list(range(N_CLASSES)))
    cm = confusion_matrix(labels, preds, labels=list(range(N_CLASSES)))
    elapsed = time.time() - t0

    print(f"  Test: acc={acc:.4f}  macroF1={macro_f1:.4f}  balAcc={bal_acc:.4f}  auc={auc:.4f}")
    return {
        "label": label,
        "Accuracy": round(acc, 4),
        "Macro-F1": round(macro_f1, 4),
        "Balanced Acc": round(bal_acc, 4),
        "AUC": round(auc, 4),
        "Log-Loss": round(ll, 4),
        "Parameters": n_params,
        "Train time(min)": round(elapsed / 60, 2),
        "train_losses": train_losses,
        "val_accs": val_accs,
        "confusion_matrix": cm,
        "config": config,
    }


# ==============================================================================
# PLOTS
# ==============================================================================
def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig); print(f"  [PLOT] -> {name}")


def plot_experiment(results, exp_name, variable):
    labels = [r["label"] for r in results]
    accs   = [r["Accuracy"] for r in results]
    f1s    = [r["Macro-F1"] for r in results]
    aucs   = [r["AUC"] for r in results]
    x = np.arange(len(labels))
    colors = plt.cm.Set2(np.linspace(0, 1, max(3, len(labels))))[:len(labels)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(f"Experiment: {variable} (HAM10000, ResNet50)", fontsize=13, fontweight="bold")
    for ax, vals, title in [(axes[0], accs, "Test Accuracy"),
                            (axes[1], f1s, "Macro-F1"),
                            (axes[2], aucs, "macro-AUC")]:
        bars = ax.bar(x, vals, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right")
        ax.set_ylim(0, 1.12); ax.set_title(title)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    _save(fig, f"HAM_exp_{exp_name}_bars.png")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{variable} — Learning Curves (HAM10000)", fontsize=12, fontweight="bold")
    epochs = range(1, N_EPOCHS + 1)
    for r, c in zip(results, colors):
        if isinstance(r.get("train_losses"), list) and len(r["train_losses"]) == N_EPOCHS:
            axes[0].plot(epochs, r["train_losses"], "o-", label=r["label"], color=c)
            axes[1].plot(epochs, r["val_accs"], "s-", label=r["label"], color=c)
    axes[0].set_title("Training Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].set_title("Validation Accuracy"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    for ax in axes: ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)
    _save(fig, f"HAM_exp_{exp_name}_curves.png")

    best = max(results, key=lambda r: r["Macro-F1"])
    if isinstance(best.get("confusion_matrix"), np.ndarray):
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(best["confusion_matrix"].astype(int), annot=True, fmt="d",
                    cmap="Blues", xticklabels=DX_CLASSES, yticklabels=DX_CLASSES, ax=ax)
        ax.set_title(f"Confusion — Best {variable}: {best['label']}", fontweight="bold")
        ax.set_ylabel("True"); ax.set_xlabel("Predicted")
        _save(fig, f"HAM_exp_{exp_name}_confusion_best.png")


def plot_final_summary(all_best):
    rows = [{"Experiment": i["variable"], "Best Value": i["best_value"],
             "Best Accuracy": i["best_acc"], "Best Macro-F1": i["best_f1"],
             "Best AUC": i["best_auc"]} for i in all_best.values()]
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "HAM_best_hyperparameters.csv"), index=False)
    fig, ax = plt.subplots(figsize=(13, max(4, len(rows) * 0.7 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=df.values, colLabels=df.columns, cellLoc="center",
                   loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor("#1a3a5c"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(rows) + 1):
        for j in range(len(df.columns)):
            tbl[i, j].set_facecolor("#EEF4FF" if i % 2 == 0 else "#FFFFFF")
    fig.suptitle("Best Hyperparameter per Experiment — HAM10000 (ResNet50)",
                 fontsize=12, fontweight="bold")
    _save(fig, "HAM_best_hyperparameters_table.png")
    print("\n  Best hyperparameters per experiment:")
    print(df.to_string(index=False))
    return df


BACKBONE_PARAMS = {
    "vgg11": 132.9, "vgg13": 133.0, "vgg16": 138.4, "vgg19": 143.7,
    "resnet18": 11.7, "resnet34": 21.8, "resnet50": 25.6, "resnet101": 44.5,
    "resnet152": 60.2, "inception_v3": 27.2, "inception_v4": 42.7,
    "inception_resnet": 55.9, "unet": 31.0, "unet_pp": 36.6,
    "attention_unet": 34.9, "residual_unet": 32.1, "unet3d": 31.0,
}


def build_param_efficiency_table():
    if not os.path.exists(MAIN_TALLY):
        print(f"  [WARN] {MAIN_TALLY} not found — skipping param table"); return None
    df = pd.read_csv(MAIN_TALLY)
    df["Parameters(M)"] = df["Backbone"].map(BACKBONE_PARAMS).fillna(0)
    df = df.sort_values("Parameters(M)")
    cols = ["Backbone", "Family", "Year", "Parameters(M)", "Test Accuracy",
            "Test AUC", "WEAT |d|", "H_epistemic", "Train time (min)"]
    result = df[[c for c in cols if c in df.columns]].reset_index(drop=True)
    result.to_csv(os.path.join(OUT_DIR, "HAM_parameter_efficiency.csv"), index=False)
    print("\n  Parameter Efficiency Table:")
    print(result.to_string(index=False))
    return result


def plot_parameter_efficiency(param_df):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Parameter Efficiency Analysis (HAM10000)", fontsize=13, fontweight="bold")
    ax = axes[0]
    for fam, c in zip(param_df["Family"].unique(),
                      plt.cm.tab10(np.linspace(0, 1, param_df["Family"].nunique()))):
        sub = param_df[param_df["Family"] == fam]
        ax.scatter(sub["Parameters(M)"], sub["Test Accuracy"], label=fam, color=c, s=80, zorder=3)
        for _, row in sub.iterrows():
            ax.annotate(row["Backbone"], (row["Parameters(M)"], row["Test Accuracy"]),
                        textcoords="offset points", xytext=(5, 3), fontsize=7)
    ax.set_xlabel("Parameters (Millions)"); ax.set_ylabel("Test Accuracy")
    ax.set_title("Parameters vs Accuracy"); ax.legend(fontsize=8); ax.grid(ls="--", alpha=0.4)
    ax = axes[1]
    if "WEAT |d|" in param_df.columns:
        colors = plt.cm.RdYlGn_r(np.linspace(0, 1, len(param_df)))
        ax.bar(range(len(param_df)), param_df["WEAT |d|"], color=colors,
               edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(param_df)))
        ax.set_xticklabels(param_df["Backbone"], rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("WEAT Bias |d|"); ax.set_title("Backbone vs Representational Bias")
        ax.axhline(0.5, color="red", ls="--", lw=1, label="High bias threshold")
        ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
        for i, (_, row) in enumerate(param_df.iterrows()):
            ax.text(i, 0.02, f"{row['Parameters(M)']:.0f}M", ha="center",
                    va="bottom", fontsize=6, rotation=90)
    _save(fig, "HAM_parameter_efficiency.png")


# ==============================================================================
# MAIN  (resume-from-incremental)
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("  HYPERPARAMETER STUDY — HAM10000  (ResNet50, 8 epochs)")
    print("=" * 60)

    img_index = build_image_index()
    split_df  = load_and_split_csv(CSV_PATH, img_index)

    all_best = {}
    all_results_flat = []
    inc_csv = os.path.join(OUT_DIR, "HAM_hyperparam_incremental.csv")
    completed_labels = set()
    old_df = None

    if os.path.exists(inc_csv):
        old_df = pd.read_csv(inc_csv)
        print(f"\n[RESUME] {len(old_df)} configs already done")
        for _, row in old_df.iterrows():
            all_results_flat.append(row.to_dict())
            completed_labels.add(f"{row['experiment']}::{row['label']}")
        for exp_name, exp_cfg in EXPERIMENTS.items():
            rows = old_df[old_df["experiment"] == exp_cfg["variable"]]
            if len(rows) >= len(exp_cfg["values"]):
                best = rows.loc[rows["Accuracy"].idxmax()]
                all_best[exp_name] = {
                    "variable": exp_cfg["variable"], "best_value": str(best["label"]),
                    "best_acc": best["Accuracy"], "best_f1": best.get("Macro-F1", "N/A"),
                    "best_auc": best.get("AUC", "N/A")}
                print(f"  \u2713 {exp_cfg['variable']} complete (best={best['label']})")

    for exp_name, exp_cfg in EXPERIMENTS.items():
        variable, fixed, values = exp_cfg["variable"], exp_cfg["fixed"], exp_cfg["values"]
        if exp_name in all_best:
            print(f"\n  SKIP {variable} — already complete"); continue
        print(f"\n{'='*60}\n  EXPERIMENT: {variable}\n  Fixed: {fixed}\n  Values: {values}\n{'='*60}")
        exp_results = []
        for val in values:
            config = dict(fixed); config[exp_name] = val
            label = str(val)
            skip_key = f"{variable}::{label}"
            if skip_key in completed_labels:
                print(f"  SKIP {variable}={label} — done")
                if old_df is not None:
                    orow = old_df[(old_df["experiment"] == variable) & (old_df["label"].astype(str) == label)]
                    if len(orow): exp_results.append(orow.iloc[0].to_dict())
                continue
            loaders = get_loaders(config.get("batch_size", 16), split_df)
            result = train_and_eval(config, loaders, label)
            result["experiment"] = variable
            exp_results.append(result); all_results_flat.append(result)
            completed_labels.add(skip_key)
            pd.DataFrame([{k: v for k, v in r.items()
                           if k not in ("train_losses", "val_accs", "confusion_matrix", "config")}
                          for r in all_results_flat]).to_csv(inc_csv, index=False)
            print(f"  [SAVED] incremental -> {inc_csv}")

        real = [r for r in exp_results if isinstance(r.get("train_losses"), list)
                and len(r.get("train_losses", [])) == N_EPOCHS]
        if real:
            plot_experiment(real, exp_name, variable)
        scorable = [r for r in exp_results if isinstance(r.get("Accuracy"), (int, float))]
        if scorable:
            best = max(scorable, key=lambda r: (r["Accuracy"], r.get("Macro-F1", 0)))
            all_best[exp_name] = {
                "variable": variable, "best_value": best["label"],
                "best_acc": best["Accuracy"], "best_f1": best.get("Macro-F1", "N/A"),
                "best_auc": best.get("AUC", "N/A")}
            print(f"\n  Best {variable}: {best['label']} (acc={best['Accuracy']})")

    print(f"\n{'='*60}\n  EXPERIMENT 8: Parameter Efficiency\n{'='*60}")
    param_df = build_param_efficiency_table()
    if param_df is not None:
        plot_parameter_efficiency(param_df)

    plot_final_summary(all_best)

    pd.DataFrame([{k: v for k, v in r.items()
                   if k not in ("train_losses", "val_accs", "confusion_matrix", "config")}
                  for r in all_results_flat]).to_csv(
        os.path.join(OUT_DIR, "HAM_all_hyperparam_results.csv"), index=False)

    print("\n" + "=" * 60 + "\n  ANSWERS TO SUPERVISOR'S QUESTIONS\n" + "=" * 60)
    qmap = [("optimizer", "Q1 optimizer"), ("learning_rate", "Q2 learning rate"),
            ("batch_size", "Q3 batch size"), ("dropout", "Q4 dropout"),
            ("activation", "Q5 activation"), ("scheduler", "Q6 scheduler"),
            ("weight_decay", "Q7 weight decay")]
    for key, q in qmap:
        b = all_best.get(key, {})
        print(f"  {q}: {b.get('best_value','N/A')} "
              f"(acc={b.get('best_acc','N/A')}  macroF1={b.get('best_f1','N/A')})")
    if param_df is not None and "Parameters(M)" in param_df.columns:
        me = param_df.loc[param_df["Parameters(M)"].idxmin()]
        print(f"  Q8 efficiency: {me['Backbone']} reaches acc={me['Test Accuracy']} "
              f"with {me['Parameters(M)']}M params")

    print(f"\n[DONE] -> {OUT_DIR}")


if __name__ == "__main__":
    main()