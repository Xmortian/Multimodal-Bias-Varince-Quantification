

import os, glob, time, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (log_loss, accuracy_score, roc_auc_score,
                             f1_score, balanced_accuracy_score)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==============================================================================
# CONFIG  — must match ham_pipeline.py
# ==============================================================================
HAM_ROOT   = r"D:\SacroMri\HAM1000"
CSV_PATH   = os.path.join(HAM_ROOT, "HAM10000_metadata.csv")
IMG_DIRS   = [os.path.join(HAM_ROOT, "HAM10000_images_part_1"),
              os.path.join(HAM_ROOT, "HAM10000_images_part_2")]
OUT_DIR    = r"D:\SacroMri\HAM_bvq_outputs"
EMB_DIR    = os.path.join(OUT_DIR, "embeddings_cache")
os.makedirs(EMB_DIR, exist_ok=True)

INCREMENTAL_CSV = os.path.join(OUT_DIR, "HAM_tally_incremental.csv")
RESCORED_CSV    = os.path.join(OUT_DIR, "HAM_tally_all17_RESCORED.csv")

DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_DEFAULT   = 340
IMG_SIZE_INCEPTION = 299
EMBED_DIM          = 512
BATCH_SIZE         = 16
SEED               = 42
DELTA_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPLICATES       = 30
N_CLASSES          = 7

DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_TO_IDX  = {c: i for i, c in enumerate(DX_CLASSES)}

LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "unknown", "upper extremity",
]
LOC_TO_IDX = {c: i for i, c in enumerate(LOCALIZATIONS)}
META_DIM = 2 + len(LOCALIZATIONS)

torch.manual_seed(SEED)
np.random.seed(SEED)

BACKBONES = [
    dict(key="vgg11",            timm_name="vgg11",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg13",            timm_name="vgg13",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg16",            timm_name="vgg16",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg19",            timm_name="vgg19",               family="VGG",       year=2014, is_unet=False),
    dict(key="resnet18",         timm_name="resnet18",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet34",         timm_name="resnet34",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet50",         timm_name="resnet50",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet101",        timm_name="resnet101",           family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet152",        timm_name="resnet152",           family="ResNet",    year=2015, is_unet=False),
    dict(key="inception_v3",     timm_name="inception_v3",        family="Inception", year=2015, is_unet=False),
    dict(key="inception_v4",     timm_name="inception_v4",        family="Inception", year=2016, is_unet=False),
    dict(key="inception_resnet", timm_name="inception_resnet_v2", family="Inception", year=2016, is_unet=False),
    dict(key="unet",             timm_name=None, family="U-Net", year=2015, is_unet=True, unet_variant="basic"),
    dict(key="unet_pp",          timm_name=None, family="U-Net", year=2018, is_unet=True, unet_variant="plusplus"),
    dict(key="attention_unet",   timm_name=None, family="U-Net", year=2018, is_unet=True, unet_variant="attention"),
    dict(key="residual_unet",    timm_name=None, family="U-Net", year=2017, is_unet=True, unet_variant="residual"),
    dict(key="unet3d",           timm_name=None, family="U-Net", year=2016, is_unet=True, unet_variant="3d"),
]
CFG_BY_KEY = {c["key"]: c for c in BACKBONES}

# ==============================================================================
# DATA  (identical logic to ham_pipeline.py so splits/embeddings match exactly)
# ==============================================================================
def build_image_index():
    idx = {}
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.jpg")):
            iid = os.path.splitext(os.path.basename(p))[0]
            idx[iid] = p
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
    j = LOC_TO_IDX.get(loc, LOC_TO_IDX["unknown"])
    vec[2 + j] = 1.0
    return vec


def load_and_split_csv(csv_path, img_index):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    df["path"] = df["image_id"].map(img_index)
    df = df[df["path"].notna()].reset_index(drop=True)
    df["label"] = df["dx"].astype(str).str.strip().str.lower().map(DX_TO_IDX)
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    sx = df["sex"].astype(str).str.strip().str.lower()
    df["sex_group"] = np.where(sx == "female", 1,
                        np.where(sx == "male", 0, -1)).astype(int)
    age_num = pd.to_numeric(df["age"], errors="coerce")
    age_mean, age_std = float(age_num.mean()), float(age_num.std())
    df["meta_vec"] = df.apply(lambda r: encode_meta_row(r, age_mean, age_std), axis=1)

    lesions = (df.groupby("lesion_id")["label"]
                 .agg(lambda s: s.value_counts().index[0]).reset_index())
    les_train, les_tmp = train_test_split(
        lesions, test_size=0.20, random_state=SEED, stratify=lesions["label"])
    les_val, les_test = train_test_split(
        les_tmp, test_size=0.50, random_state=SEED, stratify=les_tmp["label"])
    split_map = {}
    for lid in les_train["lesion_id"]: split_map[lid] = "train"
    for lid in les_val["lesion_id"]:   split_map[lid] = "val"
    for lid in les_test["lesion_id"]:  split_map[lid] = "test"
    df["split"] = df["lesion_id"].map(split_map)
    return df


class DermPreprocessor:
    def __init__(self, target_size): self.target_size = target_size
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
    def __init__(self, split_df, split, img_size):
        self.prep = DermPreprocessor(img_size)
        sub = split_df[split_df["split"] == split].reset_index(drop=True)
        self.records = [{
            "path": r["path"], "label": int(r["label"]),
            "sex_group": int(r["sex_group"]),
            "meta": r["meta_vec"].astype(np.float32),
        } for _, r in sub.iterrows()]
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = self.prep.process(Image.open(rec["path"]))
        except Exception:
            img = torch.zeros(3, self.prep.target_size, self.prep.target_size)
        return {"image": img,
                "label": torch.tensor(rec["label"], dtype=torch.long),
                "meta_vec": torch.from_numpy(rec["meta"]),
                "sex_group": torch.tensor(rec["sex_group"], dtype=torch.long)}


# ==============================================================================
# MODEL  (identical architecture to ham_pipeline.py so weights load cleanly)
# ==============================================================================
class UniversalImageEncoder(nn.Module):
    def __init__(self, backbone_cfg, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.key = backbone_cfg["key"]; self.is_unet = backbone_cfg.get("is_unet", False)
        if self.is_unet:
            variant = backbone_cfg.get("unet_variant", "basic")
            channels = [3, 64, 128, 256, 512]
            self.enc1, self.enc2, self.enc3, self.enc4 = self._make_encoder(channels, variant)
            self.pool_op = nn.MaxPool2d(2); feat_dim = channels[-1]
            if variant == "attention":
                from torch.nn import Conv2d, BatchNorm2d, Sigmoid
                self.ag = nn.Sequential(Conv2d(feat_dim + channels[-2], 1, 1), BatchNorm2d(1), Sigmoid())
            if variant == "plusplus":
                self.x01 = self._double_conv(channels[1] + channels[2], channels[1])
                self.x11 = self._double_conv(channels[2] + channels[3], channels[2])
        else:
            bb = timm.create_model(backbone_cfg["timm_name"], pretrained=True,
                                   num_classes=0, global_pool="")
            self.backbone = bb
            with torch.no_grad():
                bb.eval()
                ds = IMG_SIZE_INCEPTION if backbone_cfg["key"] == "inception_v3" else IMG_SIZE_DEFAULT
                feat_dim = bb(torch.zeros(1, 3, ds, ds)).shape[1]
        self.gap = nn.AdaptiveAvgPool2d(1)
        mid = max(embed_dim, feat_dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, mid), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(mid, embed_dim), nn.Dropout(dropout_p))
    @staticmethod
    def _double_conv(in_ch, out_ch, residual=False):
        if residual: return _ResDoubleConv(in_ch, out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True))
    def _make_encoder(self, ch, variant):
        res = (variant == "residual")
        return (self._double_conv(ch[0], ch[1], res), self._double_conv(ch[1], ch[2], res),
                self._double_conv(ch[2], ch[3], res), self._double_conv(ch[3], ch[4], res))
    def _unet_forward(self, x):
        e1 = self.enc1(x); e2 = self.enc2(self.pool_op(e1))
        e3 = self.enc3(self.pool_op(e2)); e4 = self.enc4(self.pool_op(e3))
        return e4
    def forward(self, x):
        feat_map = self._unet_forward(x) if self.is_unet else self.backbone(x)
        pooled = self.gap(feat_map).flatten(1)
        return {"embedding": F.normalize(self.proj(pooled), p=2, dim=-1)}


class _ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x) + self.skip(x))


class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=META_DIM, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(64, embed_dim), nn.Dropout(dropout_p))
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)


class ExplicitCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4):
        super().__init__()
        self.h = n_heads; self.dk = embed_dim // n_heads
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln = nn.LayerNorm(embed_dim)
    def forward(self, Ei, Em):
        B, D = Ei.shape; h, dk = self.h, self.dk
        Q = self.W_Q(Em).view(B, 1, h, dk).transpose(1, 2)
        K = self.W_K(Ei).view(B, 1, h, dk).transpose(1, 2)
        V = self.W_V(Ei).view(B, 1, h, dk).transpose(1, 2)
        A = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) / dk ** 0.5, dim=-1)
        ctx = torch.matmul(A, V).transpose(1, 2).contiguous().view(B, D)
        return self.ln(self.W_O(ctx) + Em)


class MultimodalClassifier(nn.Module):
    def __init__(self, backbone_cfg, embed_dim=EMBED_DIM, n_cls=N_CLASSES, dp=0.3):
        super().__init__()
        self.img_enc = UniversalImageEncoder(backbone_cfg, embed_dim, dp)
        self.meta_enc = ClinicalEncoder(META_DIM, embed_dim, dp)
        self.attn = ExplicitCrossAttention(embed_dim, n_heads=4)
        self.clf = nn.Sequential(nn.Linear(embed_dim * 2, 256), nn.GELU(),
                                 nn.Dropout(dp), nn.Linear(256, n_cls))
    def forward(self, images, meta):
        Ei = self.img_enc(images)["embedding"]
        Em = self.meta_enc(meta)
        f = self.attn(Ei, Em)
        return self.clf(torch.cat([Ei, f], dim=-1))
    @torch.no_grad()
    def embed_and_logits(self, images, meta):
        Ei = self.img_enc(images)["embedding"]
        Em = self.meta_enc(meta)
        f = self.attn(Ei, Em)
        emb = torch.cat([Ei, f], dim=-1)
        logits = self.clf(emb)
        return emb, logits


# ==============================================================================
# STRATIFIED ssCV
# ==============================================================================
class StratifiedSSCV:
    def __init__(self, delta_grid=DELTA_GRID, n_replicates=N_REPLICATES):
        self.deltas = delta_grid; self.R = n_replicates
        self.rng = np.random.default_rng(SEED)

    def _stratified_indices(self, labels, keep_frac):
        """Pick keep_frac of each class -> proportional, never drops a class."""
        idx_keep = []
        for c in np.unique(labels):
            c_idx = np.where(labels == c)[0]
            n_c = max(1, int(round(len(c_idx) * keep_frac)))  # at least 1 per class
            chosen = self.rng.choice(c_idx, size=min(n_c, len(c_idx)), replace=False)
            idx_keep.append(chosen)
        return np.concatenate(idx_keep)

    def _probe(self, Xtr, ytr, Xte, yte):
        sc = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                 random_state=int(self.rng.integers(9999)))
        clf.fit(sc.fit_transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))
        ll = log_loss(yte, prob, labels=list(range(N_CLASSES)))
        acc = accuracy_score(yte, clf.predict(sc.transform(Xte)))
        return ll, acc

    def run(self, embs_tr, labs_tr, embs_te, labs_te):
        records = []
        for delta in self.deltas:
            keep = 1.0 - delta
            for r in range(self.R):
                idx = self._stratified_indices(labs_tr, keep)
                if len(np.unique(labs_tr[idx])) < 2:
                    continue
                ll, acc = self._probe(embs_tr[idx], labs_tr[idx], embs_te, labs_te)
                records.append({"delta": delta, "replicate": r,
                                "loss": ll, "accuracy": acc, "n_train": len(idx)})
        df = pd.DataFrame(records)
        df["sigma2"] = df.groupby("delta")["loss"].transform(lambda x: x.var(ddof=1))
        return df


# ==============================================================================
# EMBEDDING (RE)GENERATION  — cached
# ==============================================================================
def get_embeddings(cfg, split_df):
    """Load cache if present; else load weights, forward-pass, cache."""
    key = cfg["key"]
    cache = os.path.join(EMB_DIR, f"HAM_emb_{key}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return (z["emb_tr"], z["lab_tr"],
                z["emb_te"], z["lab_te"], z["sex_te"],
                z["logits_te"], z["lab_te"])
    ckpt = os.path.join(OUT_DIR, f"HAM_{key}_best.pt")
    if not os.path.exists(ckpt):
        raise FileNotFoundError(f"missing checkpoint: {ckpt}")
    img_size = IMG_SIZE_INCEPTION if key == "inception_v3" else IMG_SIZE_DEFAULT

    model = MultimodalClassifier(cfg, EMBED_DIM).to(DEVICE)
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    model.eval()

    def run_split(split):
        ds = HAMDataset(split_df, split, img_size)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=0, pin_memory=True)
        embs, labs, sexes, logits = [], [], [], []
        with torch.no_grad():
            for b in dl:
                imgs = b["image"].to(DEVICE).float()
                meta = b["meta_vec"].to(DEVICE).float()
                e, lg = model.embed_and_logits(imgs, meta)
                embs.append(e.cpu().numpy()); logits.append(lg.cpu().numpy())
                labs.append(b["label"].numpy()); sexes.append(b["sex_group"].numpy())
        return (np.vstack(embs).astype(np.float32), np.concatenate(labs),
                np.concatenate(sexes), np.vstack(logits).astype(np.float32))

    emb_tr, lab_tr, _, _ = run_split("train")
    emb_te, lab_te, sex_te, logits_te = run_split("test")
    np.savez_compressed(cache, emb_tr=emb_tr, lab_tr=lab_tr,
                        emb_te=emb_te, lab_te=lab_te, sex_te=sex_te,
                        logits_te=logits_te)
    del model
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return emb_tr, lab_tr, emb_te, lab_te, sex_te, logits_te, lab_te


# ==============================================================================
# MAIN
# ==============================================================================
def main():
    print("=" * 60)
    print("  HAM10000 RE-SCORE  (macro-F1 + balanced-acc + stratified ssCV)")
    print("=" * 60)

    if not os.path.exists(INCREMENTAL_CSV):
        print(f"ERROR: {INCREMENTAL_CSV} not found. Let the main run finish first.")
        return
    tally = pd.read_csv(INCREMENTAL_CSV)
    print(f"Found {len(tally)} completed backbones to rescore: "
          f"{tally['Backbone'].tolist()}")

    img_index = build_image_index()
    split_df = load_and_split_csv(CSV_PATH, img_index)

    new_macro_f1, new_bal_acc, new_s2 = {}, {}, {"0.0": {}, "0.3": {}, "0.5": {}}

    for _, row in tally.iterrows():
        key = row["Backbone"]
        if key not in CFG_BY_KEY:
            print(f"  [skip] {key}: not in registry"); continue
        cfg = CFG_BY_KEY[key]
        print(f"\n--- rescoring {key} ---")
        t0 = time.time()
        try:
            (emb_tr, lab_tr, emb_te, lab_te,
             sex_te, logits_te, _) = get_embeddings(cfg, split_df)
        except FileNotFoundError as e:
            print(f"  [skip] {e}"); continue

        # ---- change 1: imbalance-robust test metrics ----
        preds = logits_te.argmax(1)
        macro_f1 = f1_score(lab_te, preds, average="macro",
                            labels=list(range(N_CLASSES)), zero_division=0)
        bal_acc = balanced_accuracy_score(lab_te, preds)
        new_macro_f1[key] = round(float(macro_f1), 4)
        new_bal_acc[key]  = round(float(bal_acc), 4)
        print(f"  Macro-F1={macro_f1:.4f}  Balanced-Acc={bal_acc:.4f}  "
              f"(raw acc was {row['Test Accuracy']})")

        # ---- change 2: stratified ssCV (consistent for all) ----
        sscv = StratifiedSSCV(DELTA_GRID, N_REPLICATES).run(
                   emb_tr, lab_tr, emb_te, lab_te)
        sscv.to_csv(os.path.join(OUT_DIR, f"HAM_sscv_strat_{key}.csv"), index=False)
        for d in (0.0, 0.3, 0.5):
            s = sscv[sscv["delta"] == d]["loss"]
            v = float(s.var(ddof=1)) if len(s) > 1 else float("nan")
            new_s2[f"{d}"][key] = round(v, 6)
        print(f"  stratified sigma2: d0.0={new_s2['0.0'][key]}  "
              f"d0.3={new_s2['0.3'][key]}  d0.5={new_s2['0.5'][key]}  "
              f"({(time.time()-t0)/60:.1f} min)")

    # ---- patch the table ----
    tally["Macro-F1"]          = tally["Backbone"].map(new_macro_f1)
    tally["Balanced Acc"]      = tally["Backbone"].map(new_bal_acc)
    tally["sigma2_d0.0_strat"] = tally["Backbone"].map(new_s2["0.0"])
    tally["sigma2_d0.3_strat"] = tally["Backbone"].map(new_s2["0.3"])
    tally["sigma2_d0.5_strat"] = tally["Backbone"].map(new_s2["0.5"])
    tally.to_csv(RESCORED_CSV, index=False)
    print(f"\n[SAVED] {RESCORED_CSV}")

    # ---- a couple of refreshed plots that use the new metrics ----
    _plot_rescored(tally)

    print("\n--- SUMMARY (raw acc vs balanced metrics) ---")
    cols = ["Backbone", "Test Accuracy", "Macro-F1", "Balanced Acc",
            "sigma2_d0.3", "sigma2_d0.3_strat"]
    cols = [c for c in cols if c in tally.columns]
    print(tally[cols].to_string(index=False))
    print("\n[DONE]")


def _plot_rescored(df):
    backbones = df["Backbone"].tolist()
    x = np.arange(len(backbones))
    w = 0.27

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - w, df["Test Accuracy"], w, label="Raw Accuracy", color="steelblue")
    ax.bar(x,     df["Macro-F1"],      w, label="Macro-F1", color="darkorange")
    ax.bar(x + w, df["Balanced Acc"],  w, label="Balanced Acc", color="seagreen")
    ax.axhline(1/7, color="grey", ls="--", lw=0.8, label="random (1/7)")
    ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score")
    ax.set_title("Accuracy vs Imbalance-Robust Metrics — All Backbones (HAM10000)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.savefig(os.path.join(OUT_DIR, "HAM_rescored_metrics.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  [PLOT] -> HAM_rescored_metrics.png")

    if "sigma2_d0.3" in df.columns:
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.bar(x - w/2, df["sigma2_d0.3"],       w, label="\u03c3\u00b2 random (old)", color="lightcoral")
        ax.bar(x + w/2, df["sigma2_d0.3_strat"], w, label="\u03c3\u00b2 stratified (new)", color="firebrick")
        ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Variance \u03c3\u00b2(\u03b4=0.3)")
        ax.set_title("ssCV Variance: random vs stratified subsampling (HAM10000)",
                     fontweight="bold")
        ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
        fig.savefig(os.path.join(OUT_DIR, "HAM_rescored_sscv_compare.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  [PLOT] -> HAM_rescored_sscv_compare.png")


if __name__ == "__main__":
    main()