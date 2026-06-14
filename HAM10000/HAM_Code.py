"""
=============================================================================
 MULTI-BACKBONE BIAS-VARIANCE PIPELINE — HAM10000  (all 17 backbones)
 "A Novel Procedure for Bias-Variance Quantification in Multimodal
  Learning Systems"

 Ported from the SacroMRI pipeline. Same method, same 17 backbones, same
 four quantifiers (WEAT bias, DCE, ssCV variance, MC-Dropout entropy),
 same plot suite. Only the DATA LAYER and a few task constants changed.

 KEY DIFFERENCES vs SacroMRI (why the changes were necessary):
   - HAM10000 has NO train/val/test folders and NO class subfolders.
     All 10,015 JPGs sit flat in two folders; the label is only in the CSV.
     -> We build stratified, PATIENT-LEVEL splits in code, grouped by
        lesion_id (a patient can have several images; they must not leak
        across splits).
   - 7 classes (akiec, bcc, bkl, df, mel, nv, vasc), not binary.
     -> N_CLASSES = 7; AUC computed one-vs-rest (macro).
   - Clinical features are age, sex, localization (one-hot) — REAL,
     non-leaking (confirmed by the sanity check: tabular-only acc 0.697
     vs 0.670 baseline).
   - Images are dermatoscopic RGB photos, not MRIs. The MRI-specific
     N4 bias-field approximation and centre-of-mass alignment are removed;
     we keep z-scoring + resize. (Those MRI steps were meaningless on
     dermatoscopy and could distort colour information.)
   - WEAT bias is now grounded on a REAL protected attribute (sex:
     female vs male) instead of random Gaussian poles. This makes the
     bias number defensible in the thesis.

 RESUMABILITY: after each backbone finishes, its row is appended to
   HAM_tally_incremental.csv. On restart, any backbone already present in
   that file is SKIPPED, so an interrupted run resumes where it stopped.
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

import timm
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score

from scipy.stats import permutation_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
print("\u2713  All imports successful")
print("\u2713  Script version: HAM10000-port-v1")

# ==============================================================================
# PATHS  — EDIT HERE IF YOUR FOLDERS MOVE
# ==============================================================================
HAM_ROOT   = r"D:\SacroMri\HAM1000"                       # folder from your screenshot
CSV_PATH   = os.path.join(HAM_ROOT, "HAM10000_metadata.csv")
IMG_DIRS   = [os.path.join(HAM_ROOT, "HAM10000_images_part_1"),
              os.path.join(HAM_ROOT, "HAM10000_images_part_2")]
OUT_DIR    = r"D:\SacroMri\HAM_bvq_outputs"               # all outputs start with HAM
os.makedirs(OUT_DIR, exist_ok=True)

INCREMENTAL_CSV = os.path.join(OUT_DIR, "HAM_tally_incremental.csv")
FAILED_CSV      = os.path.join(OUT_DIR, "HAM_failed.csv")
FINAL_CSV       = os.path.join(OUT_DIR, "HAM_tally_all17_FINAL.csv")

print(f"CSV  : {os.path.exists(CSV_PATH)}  -> {CSV_PATH}")
for d in IMG_DIRS:
    print(f"IMG  : {os.path.isdir(d)} -> {d}")

# ==============================================================================
# CONFIG
# ==============================================================================
DEVICE             = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_DEFAULT   = 340
IMG_SIZE_INCEPTION = 299
EMBED_DIM          = 512
BATCH_SIZE         = 16
SEED               = 42
MC_PASSES          = 50
DELTA_GRID         = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPLICATES       = 30
N_EPOCHS           = 8
N_CLASSES          = 7      # HAM10000: akiec, bcc, bkl, df, mel, nv, vasc

# 7-class diagnosis label mapping (alphabetical, matches the sanity-check output)
DX_CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DX_TO_IDX  = {c: i for i, c in enumerate(DX_CLASSES)}

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"\u2713  device={DEVICE}  embed={EMBED_DIM}  n_classes={N_CLASSES}")

# ==============================================================================
# CLINICAL FEATURES — real, non-leaking (verified by dataset_sanity_check)
#   age          : continuous, z-scored
#   sex          : female=1 / male=0 (unknown -> 0.5)
#   localization : one-hot over the body sites present in the CSV
# These feed the ClinicalEncoder exactly like SacroMRI's 6 features did.
# ==============================================================================
# localization categories are discovered at parse time and frozen here:
LOCALIZATIONS = [
    "abdomen", "acral", "back", "chest", "ear", "face", "foot",
    "genital", "hand", "lower extremity", "neck", "scalp", "trunk",
    "unknown", "upper extremity",
]
LOC_TO_IDX = {c: i for i, c in enumerate(LOCALIZATIONS)}

# meta vector = [age_z, sex, one-hot(localization)]
META_DIM = 2 + len(LOCALIZATIONS)
print(f"\u2713  Clinical meta dim = {META_DIM}  (age + sex + {len(LOCALIZATIONS)} localizations)")

# For the bias quantifier we need the protected attribute (sex) per sample.
# We carry it alongside the label so WEAT can split female vs male.

# ==============================================================================
# BACKBONE REGISTRY — all 17 (unchanged from SacroMRI)
# ==============================================================================
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

# ==============================================================================
# CSV PARSER + PATIENT-LEVEL SPLIT BUILDER
# ==============================================================================
def build_image_index():
    """Map image_id -> absolute path across both image folders."""
    idx = {}
    for d in IMG_DIRS:
        if not os.path.isdir(d):
            continue
        for p in glob.glob(os.path.join(d, "*.jpg")):
            iid = os.path.splitext(os.path.basename(p))[0]
            idx[iid] = p
    print(f"[IMG] indexed {len(idx)} jpg files across {len(IMG_DIRS)} folders")
    return idx


def encode_meta_row(row, age_mean, age_std):
    """Build the [age_z, sex, one-hot(loc)] float32 vector for one row."""
    vec = np.zeros(META_DIM, dtype=np.float32)
    # age (z-scored; missing -> 0 = mean)
    age = row.get("age", np.nan)
    try:
        age = float(age)
    except (TypeError, ValueError):
        age = np.nan
    vec[0] = 0.0 if (age != age) else (age - age_mean) / (age_std + 1e-8)
    # sex: female=1, male=0, unknown=0.5
    sex = str(row.get("sex", "")).strip().lower()
    vec[1] = 1.0 if sex == "female" else (0.0 if sex == "male" else 0.5)
    # localization one-hot
    loc = str(row.get("localization", "")).strip().lower()
    j = LOC_TO_IDX.get(loc, LOC_TO_IDX["unknown"])
    vec[2 + j] = 1.0
    return vec


def load_and_split_csv(csv_path, img_index):
    """
    Returns a dataframe with columns:
       image_id, path, label (0..6), sex_group (0=male,1=female,-1=unknown),
       meta_vec (np.float32[META_DIM]), split ('train'/'val'/'test')

    Split is stratified by dx and grouped by lesion_id so the same patient
    never appears in two splits.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower() for c in df.columns]
    # keep only rows whose image we actually have on disk
    df["path"] = df["image_id"].map(img_index)
    n_before = len(df)
    df = df[df["path"].notna()].reset_index(drop=True)
    print(f"[CSV] {n_before} rows in metadata; {len(df)} have a matching image")

    # label
    df["label"] = df["dx"].astype(str).str.strip().str.lower().map(DX_TO_IDX)
    df = df[df["label"].notna()].reset_index(drop=True)
    df["label"] = df["label"].astype(int)

    # protected attribute group for WEAT (sex)
    sx = df["sex"].astype(str).str.strip().str.lower()
    df["sex_group"] = np.where(sx == "female", 1,
                        np.where(sx == "male", 0, -1)).astype(int)

    # meta vector
    age_num = pd.to_numeric(df["age"], errors="coerce")
    age_mean, age_std = float(age_num.mean()), float(age_num.std())
    df["meta_vec"] = df.apply(
        lambda r: encode_meta_row(r, age_mean, age_std), axis=1)

    # ---- patient-level stratified split (80/10/10) ----
    # one row per lesion to decide the split, then propagate to all its images
    lesions = (df.groupby("lesion_id")["label"]
                 .agg(lambda s: s.value_counts().index[0])   # majority label per lesion
                 .reset_index())
    les_train, les_tmp = train_test_split(
        lesions, test_size=0.20, random_state=SEED,
        stratify=lesions["label"])
    les_val, les_test = train_test_split(
        les_tmp, test_size=0.50, random_state=SEED,
        stratify=les_tmp["label"])
    split_map = {}
    for lid in les_train["lesion_id"]: split_map[lid] = "train"
    for lid in les_val["lesion_id"]:   split_map[lid] = "val"
    for lid in les_test["lesion_id"]:  split_map[lid] = "test"
    df["split"] = df["lesion_id"].map(split_map)

    for s in ("train", "val", "test"):
        sub = df[df["split"] == s]
        print(f"  [{s:5s}] {len(sub):5d} images  "
              f"class counts={np.bincount(sub['label'], minlength=N_CLASSES).tolist()}")
    return df


# ==============================================================================
# IMAGE PREPROCESSOR  (dermatoscopy RGB — MRI-specific steps removed)
# ==============================================================================
class DermPreprocessor:
    def __init__(self, target_size: int):
        self.target_size = target_size

    @staticmethod
    def _zscore(arr):
        mu, sd = arr.mean(), arr.std()
        return ((arr - mu) / (sd + 1e-8)).astype(np.float32)

    def process(self, pil_img):
        # keep RGB (dermatoscopy colour matters), z-score per-channel, resize
        arr = np.asarray(pil_img.convert("RGB"), dtype=np.float32) / 255.0   # H,W,3
        arr = np.transpose(arr, (2, 0, 1))                                   # 3,H,W
        for c in range(3):
            arr[c] = self._zscore(arr[c])
        t = torch.from_numpy(arr).unsqueeze(0)                               # 1,3,H,W
        t = F.interpolate(t, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)
        return t.squeeze(0)                                                  # 3,H,W


# ==============================================================================
# DATASET
# ==============================================================================
class HAMDataset(Dataset):
    def __init__(self, split_df, split, augment=False, img_size=IMG_SIZE_DEFAULT):
        self.prep = DermPreprocessor(img_size)
        sub = split_df[split_df["split"] == split].reset_index(drop=True)
        self.records = []
        for _, r in sub.iterrows():
            self.records.append({
                "path":      r["path"],
                "label":     int(r["label"]),
                "image_id":  r["image_id"],
                "sex_group": int(r["sex_group"]),
                "meta":      r["meta_vec"].astype(np.float32),
            })
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
        ]) if augment else None
        print(f"  [{split:5s}] {len(self.records):5d} images  img_size={img_size}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:
            img = self.prep.process(Image.open(rec["path"]))
        except Exception:
            img = torch.zeros(3, self.prep.target_size,
                              self.prep.target_size, dtype=torch.float32)
        if self.aug: img = self.aug(img)
        return {"image":     img,
                "label":     torch.tensor(rec["label"], dtype=torch.long),
                "meta_vec":  torch.from_numpy(rec["meta"]),
                "sex_group": torch.tensor(rec["sex_group"], dtype=torch.long),
                "image_id":  rec["image_id"]}


def build_dataloaders(split_df, batch_size=BATCH_SIZE, img_size=IMG_SIZE_DEFAULT):
    loaders = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        ds = HAMDataset(split_df, split, augment=aug, img_size=img_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split == "train"),
                                    num_workers=0, pin_memory=True,
                                    drop_last=(split == "train"))
    return loaders


# ==============================================================================
# UNIVERSAL IMAGE ENCODER  (unchanged — GAP normalisation fix retained)
# ==============================================================================
class UniversalImageEncoder(nn.Module):
    def __init__(self, backbone_cfg, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.key     = backbone_cfg["key"]
        self.is_unet = backbone_cfg.get("is_unet", False)

        if self.is_unet:
            variant  = backbone_cfg.get("unet_variant", "basic")
            channels = [3, 64, 128, 256, 512]
            self.enc1, self.enc2, self.enc3, self.enc4 = \
                self._make_encoder(channels, variant)
            self.pool_op = nn.MaxPool2d(2)
            feat_dim     = channels[-1]
            if variant == "attention":
                from torch.nn import Conv2d, BatchNorm2d, Sigmoid
                self.ag = nn.Sequential(
                    Conv2d(feat_dim + channels[-2], 1, 1),
                    BatchNorm2d(1), Sigmoid())
            if variant == "plusplus":
                self.x01 = self._double_conv(channels[1] + channels[2], channels[1])
                self.x11 = self._double_conv(channels[2] + channels[3], channels[2])
        else:
            bb = timm.create_model(backbone_cfg["timm_name"],
                                   pretrained=True, num_classes=0,
                                   global_pool="")
            self.backbone = bb
            with torch.no_grad():
                bb.eval()
                dummy_size = (IMG_SIZE_INCEPTION
                              if backbone_cfg["key"] == "inception_v3"
                              else IMG_SIZE_DEFAULT)
                dummy = torch.zeros(1, 3, dummy_size, dummy_size)
                out = bb(dummy)
                feat_dim = out.shape[1]
            print(f"    [{self.key}] spatial feat_dim={feat_dim}")

        self.gap  = nn.AdaptiveAvgPool2d(1)
        mid       = max(embed_dim, feat_dim // 2)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, mid), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(mid, embed_dim), nn.Dropout(dropout_p),
        )
        print(f"    [{self.key}] proj: {feat_dim} -> {mid} -> {embed_dim}")

    @staticmethod
    def _double_conv(in_ch, out_ch, residual=False):
        if residual:
            return _ResDoubleConv(in_ch, out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def _make_encoder(self, ch, variant):
        res = (variant == "residual")
        return (self._double_conv(ch[0], ch[1], res),
                self._double_conv(ch[1], ch[2], res),
                self._double_conv(ch[2], ch[3], res),
                self._double_conv(ch[3], ch[4], res))

    def _unet_forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool_op(e1))
        e3 = self.enc3(self.pool_op(e2))
        e4 = self.enc4(self.pool_op(e3))
        return e4

    def forward(self, x):
        feat_map = self._unet_forward(x) if self.is_unet else self.backbone(x)
        pooled = self.gap(feat_map).flatten(1)
        embed  = self.proj(pooled)
        return {"embedding": F.normalize(embed, p=2, dim=-1)}

    @torch.no_grad()
    def embed(self, x):
        self.eval()
        return self.forward(x)["embedding"].cpu().numpy()


class _ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x) + self.skip(x))


# ==============================================================================
# CLINICAL ENCODER + CROSS-ATTENTION + CLASSIFIER
# ==============================================================================
class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=META_DIM, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(64, embed_dim), nn.Dropout(dropout_p),
        )
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)


class ExplicitCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.h = n_heads; self.dk = embed_dim // n_heads
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln  = nn.LayerNorm(embed_dim)
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
        self.img_enc  = UniversalImageEncoder(backbone_cfg, embed_dim, dp)
        self.meta_enc = ClinicalEncoder(META_DIM, embed_dim, dp)
        self.attn     = ExplicitCrossAttention(embed_dim, n_heads=4)
        self.clf      = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.GELU(),
            nn.Dropout(dp), nn.Linear(256, n_cls),
        )
    def forward(self, images, meta):
        Ei = self.img_enc(images)["embedding"]
        Em = self.meta_enc(meta)
        f  = self.attn(Ei, Em)
        return self.clf(torch.cat([Ei, f], dim=-1))
    @torch.no_grad()
    def embed(self, images, meta):
        self.eval()
        Ei = self.img_enc(images)["embedding"]
        Em = self.meta_enc(meta)
        return torch.cat([Ei, self.attn(Ei, Em)], dim=-1).cpu().numpy()


# ==============================================================================
# TRAINER  (AUC -> macro one-vs-rest for 7 classes)
# ==============================================================================
class Trainer:
    def __init__(self, model, device, lr=3e-4, label="model"):
        self.model = model.to(device); self.device = device; self.label = label
        self.opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        self.sched = torch.optim.lr_scheduler.CosineAnnealingLR(self.opt, T_max=N_EPOCHS)
        self.loss_fn = nn.CrossEntropyLoss(); self.history = []

    def train_epoch(self, loader):
        self.model.train(); total = 0.0
        for b in loader:
            imgs = b["image"].to(self.device).float()
            meta = b["meta_vec"].to(self.device).float()
            labs = b["label"].to(self.device)
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(imgs, meta), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step(); total += loss.item()
        self.sched.step()
        return total / max(len(loader), 1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval(); all_logits, all_labs = [], []
        for b in loader:
            imgs = b["image"].to(self.device).float()
            meta = b["meta_vec"].to(self.device).float()
            all_logits.append(self.model(imgs, meta).cpu())
            all_labs.append(b["label"])
        logits = torch.cat(all_logits); labels = torch.cat(all_labs).numpy()
        probs  = F.softmax(logits, dim=-1).numpy()
        probs_c = np.clip(probs, 1e-10, 1 - 1e-10)
        probs_c = probs_c / probs_c.sum(axis=1, keepdims=True)
        # macro one-vs-rest AUC across the classes actually present
        try:
            present = np.unique(labels)
            if len(present) > 1:
                auc = roc_auc_score(labels, probs_c, multi_class="ovr",
                                    average="macro", labels=list(range(N_CLASSES)))
            else:
                auc = 0.5
        except Exception:
            auc = 0.5
        return {"accuracy": accuracy_score(labels, probs.argmax(1)),
                "log_loss": log_loss(labels, probs_c, labels=list(range(N_CLASSES))),
                "auc": auc}

    def fit(self, train_loader, val_loader):
        print(f"\n  -- {self.label} --"); best = 0.0
        for ep in range(1, N_EPOCHS + 1):
            tr_l = self.train_epoch(train_loader)
            val  = self.evaluate(val_loader)
            self.history.append({"epoch": ep, "train_loss": tr_l, **val})
            print(f"    ep {ep:02d}/{N_EPOCHS}  loss={tr_l:.4f}  "
                  f"val_acc={val['accuracy']:.4f}  auc={val['auc']:.4f}")
            if val["accuracy"] > best:
                best = val["accuracy"]
                torch.save(self.model.state_dict(),
                           os.path.join(OUT_DIR, f"HAM_{self.label}_best.pt"))
        print(f"    Best val acc: {best:.4f}")

    @torch.no_grad()
    def extract_embeddings(self, loader):
        self.model.eval(); embs, labs, sexes = [], [], []
        for b in loader:
            imgs = b["image"].to(self.device).float()
            meta = b["meta_vec"].to(self.device).float()
            embs.append(self.model.embed(imgs, meta))
            labs.append(b["label"].numpy())
            sexes.append(b["sex_group"].numpy())
        return (np.vstack(embs).astype(np.float32),
                np.concatenate(labs),
                np.concatenate(sexes))


# ==============================================================================
# BIAS & VARIANCE QUANTIFIERS
# ==============================================================================
def make_valence_attributes(embed_dim):
    """
    Kept for API symmetry, but for HAM10000 we ground WEAT on the REAL
    protected attribute (sex), so these random poles are NOT used for the
    headline bias number. See SexWEATQuantifier below.
    """
    rng = np.random.default_rng(seed=0)
    poles = dict(a=rng.standard_normal(embed_dim), b=rng.standard_normal(embed_dim))
    return {k: (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
            for k, v in poles.items()}


class SexWEATQuantifier:
    """
    Effect-size of representational separation between the protected groups
    (female vs male) in embedding space. Uses the mean-difference direction
    as the association axis (a data-driven, grounded alternative to random
    valence poles). |d| interpretation thresholds match the SacroMRI script.
    """
    def compute(self, X, sex_group):
        fem = X[sex_group == 1]
        mal = X[sex_group == 0]
        if len(fem) < 2 or len(mal) < 2:
            return {"weat_d": 0.0, "p_value": 1.0, "interpretation": "Low |d|<0.2"}
        # association axis = direction between group means
        axis = fem.mean(0) - mal.mean(0)
        nrm = np.linalg.norm(axis) + 1e-8
        axis = axis / nrm
        sF = fem @ axis
        sM = mal @ axis
        all_s = np.concatenate([sF, sM])
        d = float((sF.mean() - sM.mean()) / (all_s.std() + 1e-8))
        def stat(a, b, axis=0): return a.mean(axis=axis) - b.mean(axis=axis)
        perm = permutation_test((sF, sM), stat, permutation_type="independent",
                                n_resamples=1000, alternative="two-sided",
                                random_state=SEED)
        return {"weat_d": d, "p_value": float(perm.pvalue),
                "interpretation": ("High |d|>=0.5" if abs(d) >= 0.5 else
                                   "Moderate |d|>=0.2" if abs(d) >= 0.2 else
                                   "Low |d|<0.2")}


class DCEQuantifier:
    """
    Direct causal effect of each clinical feature block on P(predicted class).
    For 7 classes we measure the effect on the model's max-class probability.
    Feature blocks: age (idx 0), sex (idx 1), localization (idx 2..).
    """
    FEATURE_BLOCKS = [("age", [0]), ("sex", [1]),
                      ("localization", list(range(2, META_DIM)))]
    def __init__(self, model, device):
        self.model = model.to(device); self.device = device
    @torch.no_grad()
    def measure(self, images, meta):
        self.model.eval()
        imgs = images.to(self.device).float()
        meta = meta.to(self.device).float()
        p_full = F.softmax(self.model(imgs, meta), dim=-1).cpu().numpy()
        pred = p_full.argmax(1)
        p_full_max = p_full[np.arange(len(pred)), pred]
        dce = {}
        for name, cols in self.FEATURE_BLOCKS:
            mcf = meta.clone()
            for c in cols:
                mcf[:, c] = 0.0
            p_cf = F.softmax(self.model(imgs, mcf), dim=-1).cpu().numpy()
            p_cf_same = p_cf[np.arange(len(pred)), pred]
            dce[name] = float((p_full_max - p_cf_same).mean())
        return dce


class ssCVQuantifier:
    def __init__(self, delta_grid=DELTA_GRID, n_replicates=N_REPLICATES):
        self.deltas = delta_grid; self.R = n_replicates
        self.rng = np.random.default_rng(SEED)
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
        assert embs_tr.dtype == np.float32, "Float32 contract violated"
        assert embs_te.shape[1] == embs_tr.shape[1], \
            f"Embedding dim mismatch: tr={embs_tr.shape} te={embs_te.shape}"
        records = []
        for delta in self.deltas:
            n_sub = max(N_CLASSES * 2, int(len(embs_tr) * (1.0 - delta)))
            for r in range(self.R):
                idx = self.rng.choice(len(embs_tr), size=n_sub, replace=False)
                # ensure at least 2 classes in the subsample
                if len(np.unique(labs_tr[idx])) < 2:
                    continue
                ll, acc = self._probe(embs_tr[idx], labs_tr[idx], embs_te, labs_te)
                records.append({"delta": delta, "replicate": r,
                                "loss": ll, "accuracy": acc, "n_train": n_sub})
        df = pd.DataFrame(records)
        df["sigma2"] = df.groupby("delta")["loss"].transform(lambda x: x.var(ddof=1))
        return df


class MCDropoutQuantifier:
    def __init__(self, model, n_passes=MC_PASSES, device=DEVICE):
        self.model = model; self.T = n_passes; self.device = device
    def _enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m, nn.Dropout): m.train()
    @torch.no_grad()
    def predict(self, images, meta):
        self.model.eval(); self._enable_dropout()
        imgs = images.to(self.device).float()
        meta = meta.to(self.device).float()
        passes = []
        for _ in range(self.T):
            passes.append(F.softmax(self.model(imgs, meta), dim=-1).cpu())
        P = torch.stack(passes, dim=0).numpy(); p_bar = P.mean(axis=0)
        H_total = -np.sum(p_bar * np.log(p_bar + 1e-10), axis=-1)
        per_H   = -np.sum(P * np.log(P + 1e-10), axis=-1)
        H_ale   = per_H.mean(axis=0); H_epi = H_total - H_ale
        KL = np.log(p_bar.shape[-1]) - H_total
        return {"H_total": H_total, "H_aleatoric": H_ale,
                "H_epistemic": H_epi, "KL_uniform": KL}


# ==============================================================================
# PER-BACKBONE PIPELINE
# ==============================================================================
def run_backbone(backbone_cfg, split_df):
    key = backbone_cfg["key"]
    img_size = (IMG_SIZE_INCEPTION if key == "inception_v3" else IMG_SIZE_DEFAULT)

    print(f"\n{'='*60}")
    print(f"  BACKBONE: {key.upper()}  [{backbone_cfg['family']} {backbone_cfg['year']}]"
          f"  img_size={img_size}")
    print(f"{'='*60}")
    t0 = time.time()

    loaders = build_dataloaders(split_df, batch_size=BATCH_SIZE, img_size=img_size)
    test_batch = next(iter(loaders["test"]))

    model   = MultimodalClassifier(backbone_cfg, EMBED_DIM)
    trainer = Trainer(model, DEVICE, label=key)
    trainer.fit(loaders["train"], loaders["val"])

    test_metrics = trainer.evaluate(loaders["test"])
    print(f"  Test  acc={test_metrics['accuracy']:.4f}  "
          f"auc={test_metrics['auc']:.4f}  ll={test_metrics['log_loss']:.4f}")

    emb_tr, lab_tr, _      = trainer.extract_embeddings(loaders["train"])
    emb_te, lab_te, sex_te = trainer.extract_embeddings(loaders["test"])
    print(f"  Embeddings: train={emb_tr.shape}  test={emb_te.shape}  dtype={emb_tr.dtype}")

    # Bias — grounded on real protected attribute (sex)
    bias = SexWEATQuantifier().compute(emb_te, sex_te)
    print(f"  WEAT(sex) d={bias['weat_d']:+.4f}  p={bias['p_value']:.4f}  {bias['interpretation']}")

    # DCE
    dce = DCEQuantifier(model, DEVICE).measure(test_batch["image"], test_batch["meta_vec"])

    # ssCV
    sscv_df = ssCVQuantifier(DELTA_GRID, N_REPLICATES).run(emb_tr, lab_tr, emb_te, lab_te)
    sscv_df.to_csv(os.path.join(OUT_DIR, f"HAM_sscv_{key}.csv"), index=False)

    # MC-Dropout
    ent = MCDropoutQuantifier(model, MC_PASSES, DEVICE).predict(
            test_batch["image"], test_batch["meta_vec"])

    elapsed = time.time() - t0
    print(f"  Done in {elapsed/60:.1f} min")

    def s2(df, d):
        s = df[df["delta"] == d]["loss"]
        v = float(s.var(ddof=1)) if len(s) > 1 else float("nan")
        if np.isnan(v):
            print(f"  WARNING: NaN variance at delta={d} for {key}!")
        return round(v, 6)

    return {
        "Backbone":         key,
        "Family":           backbone_cfg["family"],
        "Year":             backbone_cfg["year"],
        "Img Size":         img_size,
        "Test Accuracy":    round(test_metrics["accuracy"], 4),
        "Test AUC":         round(test_metrics["auc"], 4),
        "Test Log-Loss":    round(test_metrics["log_loss"], 4),
        "WEAT |d|":         round(abs(bias["weat_d"]), 4),
        "Bias p-value":     round(bias["p_value"], 4),
        "Bias class":       bias["interpretation"],
        "Mean |DCE|":       round(float(np.mean(np.abs(list(dce.values())))), 5),
        "DCE_age":          round(dce["age"], 6),
        "DCE_sex":          round(dce["sex"], 6),
        "DCE_localization": round(dce["localization"], 6),
        "sigma2_d0.0":      s2(sscv_df, 0.0),
        "sigma2_d0.3":      s2(sscv_df, 0.3),
        "sigma2_d0.5":      s2(sscv_df, 0.5),
        "H_epistemic":      round(float(ent["H_epistemic"].mean()), 5),
        "H_aleatoric":      round(float(ent["H_aleatoric"].mean()), 5),
        "KL_uniform":       round(float(ent["KL_uniform"].mean()), 5),
        "Train time (min)": round(elapsed / 60, 2),
    }


# ==============================================================================
# PLOTS  (same suite; titles say HAM10000)
# ==============================================================================
def _save(fig, name):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] -> {name}")


def plot_all(df):
    backbones = df["Backbone"].tolist()
    x = np.arange(len(backbones))
    colors = plt.cm.tab20(np.linspace(0, 1, len(backbones)))

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Test Performance — All 17 Backbones (HAM10000)",
                 fontsize=13, fontweight="bold")
    for ax, col, title in [(axes[0], "Test Accuracy", "Test Accuracy"),
                           (axes[1], "Test AUC", "Test AUC")]:
        bars = ax.bar(x, df[col], color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
        ax.set_ylim(0, 1.12); ax.set_ylabel(title); ax.set_title(title)
        ax.axhline(0.5, color="grey", ls="--", lw=0.8)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for bar, val in zip(bars, df[col]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    _save(fig, "HAM_final_accuracy_auc.png")

    fig, ax = plt.subplots(figsize=(16, 5))
    bars = ax.bar(x, df["WEAT |d|"], color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="red", ls="--", lw=1.2, label="|d|=0.5 High")
    ax.axhline(0.2, color="orange", ls="--", lw=1.0, label="|d|=0.2 Moderate")
    ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("WEAT |d| (sex)")
    ax.set_title("Representational Bias by Sex — All 17 Backbones (HAM10000)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    for bar, val in zip(bars, df["WEAT |d|"]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)
    _save(fig, "HAM_final_weat_bias.png")

    fig, ax = plt.subplots(figsize=(18, 6))
    w = 0.25
    ax.bar(x - w, df["sigma2_d0.0"], w, label="\u03c3\u00b2(\u03b4=0.0)", color="royalblue", alpha=0.85)
    ax.bar(x,     df["sigma2_d0.3"], w, label="\u03c3\u00b2(\u03b4=0.3)", color="darkorange", alpha=0.85)
    ax.bar(x + w, df["sigma2_d0.5"], w, label="\u03c3\u00b2(\u03b4=0.5)", color="firebrick", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Variance \u03c3\u00b2(\u03b4)")
    ax.set_title("ssCV Variance — All 17 Backbones (HAM10000)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    _save(fig, "HAM_final_sscv_variance.png")

    fig, ax = plt.subplots(figsize=(18, 6))
    ax.bar(x - 0.2, df["H_epistemic"], 0.35, label="Epistemic", color="purple", alpha=0.85)
    ax.bar(x + 0.2, df["H_aleatoric"], 0.35, label="Aleatoric", color="teal", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(backbones, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("MC-Dropout Entropy — All 17 Backbones (HAM10000)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", ls="--", alpha=0.4)
    _save(fig, "HAM_final_entropy.png")

    heat_cols = ["Test Accuracy", "Test AUC", "WEAT |d|",
                 "sigma2_d0.3", "H_epistemic", "Mean |DCE|"]
    heat_df = df.set_index("Backbone")[heat_cols].astype(float)
    norm_heat = (heat_df - heat_df.min()) / (heat_df.max() - heat_df.min() + 1e-8)
    fig, ax = plt.subplots(figsize=(13, max(7, len(backbones) * 0.5)))
    sns.heatmap(norm_heat, annot=heat_df.round(4), fmt="", cmap="RdYlGn",
                linewidths=0.5, ax=ax, cbar_kws={"label": "Normalised value"})
    ax.set_title("Backbone Comparison Heatmap — All 17 Models (HAM10000)",
                 fontsize=12, fontweight="bold")
    _save(fig, "HAM_final_heatmap.png")

    metrics = ["Test Accuracy", "Test AUC", "WEAT |d|", "H_epistemic", "Mean |DCE|"]
    radar_df = df[metrics].copy().astype(float)
    for col in metrics:
        mn, mx = radar_df[col].min(), radar_df[col].max()
        radar_df[col] = (radar_df[col] - mn) / (mx - mn + 1e-8)
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist() + [0]
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))
    for i, (_, row) in enumerate(radar_df.iterrows()):
        vals = row.tolist() + [row.tolist()[0]]
        ax.plot(angles, vals, "o-", linewidth=1.2, label=backbones[i],
                color=colors[i], alpha=0.8)
    ax.set_thetagrids(np.degrees(angles[:-1]), metrics, fontsize=9)
    ax.set_title("Radar — All 17 Backbones (HAM10000)", fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.15), fontsize=7)
    _save(fig, "HAM_final_radar.png")

    family_colors = {"VGG": "#FFF3CD", "ResNet": "#D1ECF1",
                     "Inception": "#D4EDDA", "U-Net": "#F8D7DA"}
    display_cols = ["Backbone", "Family", "Year", "Img Size",
                    "Test Accuracy", "Test AUC", "Test Log-Loss",
                    "WEAT |d|", "Bias class",
                    "sigma2_d0.0", "sigma2_d0.3", "sigma2_d0.5",
                    "H_epistemic", "H_aleatoric", "Train time (min)"]
    show = df[display_cols]
    fig, ax = plt.subplots(figsize=(26, max(5, len(df) * 0.65 + 1.5)))
    ax.axis("off")
    tbl = ax.table(cellText=show.values, colLabels=show.columns,
                   cellLoc="center", loc="center", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for j in range(len(show.columns)):
        tbl[0, j].set_facecolor("#1a3a5c")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, (_, row) in enumerate(df.iterrows(), start=1):
        clr = family_colors.get(row["Family"], "#FFFFFF")
        for j in range(len(show.columns)):
            tbl[i, j].set_facecolor(clr)
    fig.suptitle(
        "Bias-Variance Tally — All 17 Backbones — HAM10000\n"
        "Multimodal (dermatoscopy image + age/sex/localization). "
        "WEAT grounded on sex; 7-class diagnosis.",
        fontsize=10, fontweight="bold")
    _save(fig, "HAM_final_tally_all17.png")
    print("\n  All plots saved.")


# ==============================================================================
# MAIN  (with resume-from-incremental support)
# ==============================================================================
def main():
    print("\n" + "=" * 60)
    print("  BIAS-VARIANCE PIPELINE — HAM10000  (all 17 backbones)")
    print("=" * 60)

    img_index = build_image_index()
    split_df  = load_and_split_csv(CSV_PATH, img_index)

    # ---- RESUME: load any already-completed backbones ----
    done_keys = set()
    all_results = []
    if os.path.exists(INCREMENTAL_CSV):
        prev = pd.read_csv(INCREMENTAL_CSV)
        all_results = prev.to_dict("records")
        done_keys = set(prev["Backbone"].tolist())
        print(f"\n[RESUME] Found {len(done_keys)} completed backbones: "
              f"{sorted(done_keys)}")
        print("[RESUME] These will be skipped.")

    failed = []
    for i, cfg in enumerate(BACKBONES):
        if cfg["key"] in done_keys:
            print(f"\n[{i+1}/{len(BACKBONES)}] {cfg['key']} — already done, skipping.")
            continue
        print(f"\n[{i+1}/{len(BACKBONES)}] {cfg['key']} ...")
        try:
            result = run_backbone(cfg, split_df)
            all_results.append(result)
            # incremental save AFTER each backbone -> safe to interrupt
            pd.DataFrame(all_results).to_csv(INCREMENTAL_CSV, index=False)
            print(f"  [SAVED] incremental -> {INCREMENTAL_CSV}")
        except Exception as e:
            import traceback
            print(f"  FAILED: {cfg['key']} — {e}")
            traceback.print_exc()
            failed.append({"Backbone": cfg["key"], "Error": str(e)})
            pd.DataFrame(failed).to_csv(FAILED_CSV, index=False)
            continue

    if not all_results:
        print("No results — all backbones failed."); return

    tally = pd.DataFrame(all_results)
    order = {"VGG": 0, "ResNet": 1, "Inception": 2, "U-Net": 3}
    tally["_o"] = tally["Family"].map(order).fillna(99)
    tally = (tally.sort_values(["_o", "Year", "Backbone"])
                  .drop(columns="_o").reset_index(drop=True))
    tally.to_csv(FINAL_CSV, index=False)

    print("\n" + "=" * 60)
    print("  FINAL TABLE — ALL 17 BACKBONES (HAM10000)")
    print("=" * 60)
    disp = ["Backbone", "Family", "Year", "Img Size",
            "Test Accuracy", "Test AUC", "WEAT |d|",
            "sigma2_d0.3", "H_epistemic", "Train time (min)"]
    print(tally[disp].to_string(index=False))
    print("=" * 60)

    plot_all(tally)

    nan_cols = tally[["sigma2_d0.0", "sigma2_d0.3", "sigma2_d0.5"]].isna().sum()
    print(f"\n  NaN check — sigma2 columns:")
    for col, n in nan_cols.items():
        print(f"    {col}: {'PASS' if n == 0 else f'FAIL ({n} NaNs)'}")

    best_acc = tally.loc[tally["Test Accuracy"].idxmax()]
    low_bias = tally.loc[tally["WEAT |d|"].idxmin()]
    print(f"\n  Best Accuracy : {best_acc['Backbone']}  ({best_acc['Test Accuracy']:.4f})")
    print(f"  Lowest Bias   : {low_bias['Backbone']}  (|d|={low_bias['WEAT |d|']:.4f})")
    if failed:
        print(f"\n  Failed backbones: {[f['Backbone'] for f in failed]}")
    print(f"\n[DONE] -> {OUT_DIR}")


if __name__ == "__main__":
    main()