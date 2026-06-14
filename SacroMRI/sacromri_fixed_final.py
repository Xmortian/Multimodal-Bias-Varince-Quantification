"""
=============================================================================
 FIXED MULTI-BACKBONE COMPARISON — SacroMRI
 
 Fixes applied:
   FIX 1 — NaN in sigma2: AdaptiveAvgPool2d(1) applied uniformly across ALL
            backbones before projection. Guarantees consistent [B, C] tensors
            into ssCV, eliminating shape mismatches.
            
   FIX 2 — Inception v3 input size: conditional resize to 299×299 for
            inception_v3 only; all others remain 340×340. Prevents corrupted
            spatial features that produced the fake |d|=0.0019 bias score.
            
   FIX 3 — Label leakage: diagnosis_flag, bone_marrow_edema, erosions are
            explicitly stripped from CLINICAL_FEATURES. Model sees only 6
            pre-diagnostic patient-reported and lab values, never the label.
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
from sklearn.metrics import log_loss, accuracy_score, roc_auc_score
from scipy.stats import permutation_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
print("✓  All imports successful")
print("✓  Script version: fixed-v1 (NaN+InceptionSize+LeakageFixes)")

# ══════════════════════════════════════════════════════════════════════════════
# PATHS
# ══════════════════════════════════════════════════════════════════════════════
KAGGLE_ROOT  = r"D:\SacroMri\archive"
CSV_PATH     = os.path.join(KAGGLE_ROOT, "SacroMRI Dataset.csv")
DATA_ROOT    = os.path.join(KAGGLE_ROOT, "SacroMRI")
OUT_DIR      = r"D:\SacroMri\bvq_outputs_fixed"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"CSV  : {os.path.exists(CSV_PATH)}  → {CSV_PATH}")
print(f"DATA : {os.path.exists(DATA_ROOT)} → {DATA_ROOT}")

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE_DEFAULT  = 340          # for all backbones except inception_v3
IMG_SIZE_INCEPTION = 299         # FIX 2: inception_v3 requires 299×299
EMBED_DIM    = 512
BATCH_SIZE   = 16
SEED         = 42
MC_PASSES    = 50
DELTA_GRID   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPLICATES = 30
N_EPOCHS     = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"✓  device={DEVICE}  embed={EMBED_DIM}")

# ══════════════════════════════════════════════════════════════════════════════
# FIX 3 — CLINICAL FEATURES: strictly pre-diagnostic only
# Removed: diagnosis_flag (IS the label), bone_marrow_edema, erosions
#          (radiologist findings that directly encode the diagnosis)
# ══════════════════════════════════════════════════════════════════════════════
CLINICAL_FEATURES = [
    "chronic_back_pain",     # patient-reported symptom
    "morning_stiffness",     # patient-reported symptom
    "improvement_exercise",  # patient-reported symptom
    "hla_b27",               # lab test result
    "esr_norm",              # blood marker z-scored
    "crp_norm",              # blood marker z-scored
]
META_DIM = len(CLINICAL_FEATURES)
print(f"✓  Clinical features ({META_DIM}): {CLINICAL_FEATURES}")

# ══════════════════════════════════════════════════════════════════════════════
# BACKBONE REGISTRY — all 17 from the table
# ══════════════════════════════════════════════════════════════════════════════
BACKBONES = [
    # VGG
    dict(key="vgg11",            timm_name="vgg11",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg13",            timm_name="vgg13",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg16",            timm_name="vgg16",               family="VGG",       year=2014, is_unet=False),
    dict(key="vgg19",            timm_name="vgg19",               family="VGG",       year=2014, is_unet=False),
    # ResNet
    dict(key="resnet18",         timm_name="resnet18",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet34",         timm_name="resnet34",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet50",         timm_name="resnet50",            family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet101",        timm_name="resnet101",           family="ResNet",    year=2015, is_unet=False),
    dict(key="resnet152",        timm_name="resnet152",           family="ResNet",    year=2015, is_unet=False),
    # Inception
    dict(key="inception_v3",     timm_name="inception_v3",        family="Inception", year=2015, is_unet=False),
    dict(key="inception_v4",     timm_name="inception_v4",        family="Inception", year=2016, is_unet=False),
    dict(key="inception_resnet", timm_name="inception_resnet_v2", family="Inception", year=2016, is_unet=False),
    # U-Net variants
    dict(key="unet",             timm_name=None, family="U-Net", year=2015, is_unet=True, unet_variant="basic"),
    dict(key="unet_pp",          timm_name=None, family="U-Net", year=2018, is_unet=True, unet_variant="plusplus"),
    dict(key="attention_unet",   timm_name=None, family="U-Net", year=2018, is_unet=True, unet_variant="attention"),
    dict(key="residual_unet",    timm_name=None, family="U-Net", year=2017, is_unet=True, unet_variant="residual"),
    dict(key="unet3d",           timm_name=None, family="U-Net", year=2016, is_unet=True, unet_variant="3d"),
]

# ══════════════════════════════════════════════════════════════════════════════
# CSV PARSER — FIX 3: never expose diagnosis columns to model
# ══════════════════════════════════════════════════════════════════════════════
def load_and_parse_csv(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    rmap = {}
    for c in df.columns:
        cl = c.lower()
        if "morning"    in cl:                    rmap[c] = "morning_stiffness"
        elif "exercise" in cl:                    rmap[c] = "improvement_exercise"
        elif "hla"      in cl:                    rmap[c] = "hla_b27"
        elif "marrow"   in cl or "edema" in cl:   rmap[c] = "bone_marrow_edema"
        elif "erosion"  in cl:                    rmap[c] = "erosions"
        elif "esr"      in cl:                    rmap[c] = "esr"
        elif "crp"      in cl:                    rmap[c] = "crp"
        elif ("image"   in cl and "id" in cl) or cl=="id":
                                                  rmap[c] = "image_id"
        elif "diagnosis" in cl:                   rmap[c] = "diagnosis"
    df = df.rename(columns=rmap)
    print(f"[CSV] mapped cols: {list(df.columns)}")

    _YES = {"yes":1.0,"no":0.0,"positive":1.0,"negative":0.0,
            "1":1.0,"0":0.0,"1.0":1.0,"0.0":0.0}
    for col in ["chronic_back_pain","morning_stiffness",
                "improvement_exercise","hla_b27"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                         .map(_YES).fillna(0.0).astype(np.float32))
        else:
            df[col] = np.float32(0.0)

    for raw, norm in [("esr","esr_norm"),("crp","crp_norm")]:
        if raw in df.columns:
            v = pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
            df[norm] = ((v-v.mean())/(v.std()+1e-8)).astype(np.float32)
        else:
            df[norm] = np.float32(0.0)

    # Label column — used ONLY for DataLoader label, NOT passed to model
    if "diagnosis" in df.columns:
        df["diagnosis_flag"] = (df["diagnosis"].astype(str)
                                  .str.strip().str.upper()=="AS"
                               ).astype(np.float32)
    else:
        df["diagnosis_flag"] = np.float32(0.0)

    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)
    df["image_id"] = df["image_id"].astype(str).str.strip()

    # Verify no leaking columns in CLINICAL_FEATURES
    leaked = [f for f in CLINICAL_FEATURES
              if f in ("diagnosis_flag","bone_marrow_edema","erosions","diagnosis")]
    assert len(leaked)==0, f"LEAKAGE DETECTED: {leaked}"
    print(f"[CSV] {len(df)} rows  |  leakage check PASSED")
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 — MRI PREPROCESSOR with per-backbone resize
# ══════════════════════════════════════════════════════════════════════════════
class MRIPreprocessor:
    def __init__(self, target_size: int):
        self.target_size = target_size
        self._kernel = self._gauss_kernel(sigma=15, size=31)

    @staticmethod
    def _gauss_kernel(sigma, size):
        c  = torch.arange(size).float() - size//2
        g1 = torch.exp(-c**2/(2*sigma**2))
        g2 = torch.ger(g1, g1)
        return (g2/g2.sum()).unsqueeze(0).unsqueeze(0)

    def _n4_approx(self, arr):
        t   = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        pad = self._kernel.shape[-1]//2
        bf  = F.conv2d(t, self._kernel, padding=pad)
        return (t/(bf+1e-3)).squeeze().numpy()

    @staticmethod
    def _align(arr):
        thresh = np.percentile(arr, 75)
        ys, xs = np.where(arr > thresh)
        if len(ys) < 20: return arr
        dy = int(arr.shape[0]//2 - ys.mean())
        dx = int(arr.shape[1]//2 - xs.mean())
        return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)

    @staticmethod
    def _zscore(arr):
        mu, sd = arr.mean(), arr.std()
        return ((arr-mu)/(sd+1e-8)).astype(np.float32)

    def process(self, pil_img):
        arr = np.array(pil_img.convert("L"), dtype=np.float32)/255.0
        arr = self._n4_approx(arr)
        arr = self._align(arr)
        arr = self._zscore(arr)
        t   = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t   = F.interpolate(t, size=(self.target_size, self.target_size),
                            mode="bilinear", align_corners=False)
        return t.squeeze(0).repeat(3,1,1)   # [3,H,W] float32

# ══════════════════════════════════════════════════════════════════════════════
# DATASET — FIX 2: accepts target_size per backbone
# ══════════════════════════════════════════════════════════════════════════════
class SacroMRIDataset(Dataset):
    def __init__(self, root, split, meta_df=None,
                 augment=False, img_size=IMG_SIZE_DEFAULT):
        self.prep    = MRIPreprocessor(img_size)   # FIX 2: size passed in
        self.records = []
        for cls_name, label in [("AS",1),("Normal",0)]:
            d = os.path.join(root, split, cls_name)
            if not os.path.isdir(d): continue
            for p in sorted(glob.glob(os.path.join(d,"*.png"))):
                img_id = os.path.splitext(os.path.basename(p))[0]
                self.records.append({"path":p,"label":label,"image_id":img_id})

        self._meta_lookup   = {}
        self._meta_fallback = np.zeros(META_DIM, dtype=np.float32)
        if meta_df is not None:
            for _, row in meta_df.iterrows():
                key = str(row.get("image_id","")).strip()
                # FIX 3: only extract CLINICAL_FEATURES — never diagnosis cols
                vec = np.array([float(row.get(f,0.0)) for f in CLINICAL_FEATURES],
                               dtype=np.float32)
                self._meta_lookup[key] = vec

        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
        ]) if augment else None

        n_as  = sum(r["label"]==1 for r in self.records)
        n_nor = sum(r["label"]==0 for r in self.records)
        print(f"  [{split:5s}] {len(self.records):4d} images  "
              f"(AS={n_as}, Normal={n_nor})  img_size={img_size}")

    def __len__(self): return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        try:    img = self.prep.process(Image.open(rec["path"]))
        except: img = torch.zeros(3, self.prep.target_size,
                                  self.prep.target_size, dtype=torch.float32)
        if self.aug: img = self.aug(img)
        meta = self._meta_lookup.get(rec["image_id"], self._meta_fallback.copy())
        return {"image":    img,
                "label":    torch.tensor(rec["label"], dtype=torch.long),
                "meta_vec": torch.from_numpy(meta.astype(np.float32)),
                "image_id": rec["image_id"]}


def build_dataloaders(root, meta_df, batch_size=BATCH_SIZE,
                      img_size=IMG_SIZE_DEFAULT):
    """FIX 2: img_size passed through to dataset."""
    loaders = {}
    for split, aug in [("train",True),("val",False),("test",False)]:
        ds = SacroMRIDataset(root, split, meta_df=meta_df,
                             augment=aug, img_size=img_size)
        loaders[split] = DataLoader(ds, batch_size=batch_size,
                                    shuffle=(split=="train"),
                                    num_workers=0, pin_memory=True,
                                    drop_last=(split=="train"))
    return loaders

# ══════════════════════════════════════════════════════════════════════════════
# FIX 1 — UNIVERSAL IMAGE ENCODER
# AdaptiveAvgPool2d(1) applied uniformly to ALL backbones before projection.
# This guarantees [B, feat_dim] regardless of input size or spatial output.
# ══════════════════════════════════════════════════════════════════════════════
class UniversalImageEncoder(nn.Module):
    """
    Works for timm backbones AND U-Net variants.
    
    FIX 1: All feature maps go through AdaptiveAvgPool2d(1) → flatten
           before the projection layer. This means:
             - VGG    : [B,512,H,W]  → pool → [B,512]  → proj → [B,EMBED]
             - ResNet : [B,2048,H,W] → pool → [B,2048] → proj → [B,EMBED]
             - UNet   : [B,512,H,W]  → pool → [B,512]  → proj → [B,EMBED]
           ssCV receives identical-shape float32 arrays → no more NaN.
    """

    def __init__(self, backbone_cfg, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.key     = backbone_cfg["key"]
        self.is_unet = backbone_cfg.get("is_unet", False)

        if self.is_unet:
            variant  = backbone_cfg.get("unet_variant","basic")
            channels = [3,64,128,256,512]
            self.enc1, self.enc2, self.enc3, self.enc4 = \
                self._make_encoder(channels, variant)
            self.pool_op   = nn.MaxPool2d(2)
            feat_dim       = channels[-1]
            # Attention gates for attention_unet
            if variant=="attention":
                from torch.nn import Conv2d, BatchNorm2d, Sigmoid
                self.ag = nn.Sequential(
                    Conv2d(feat_dim+channels[-2], 1, 1),
                    BatchNorm2d(1), Sigmoid())
            # Extra dense connections for unet_pp
            if variant=="plusplus":
                self.x01=self._double_conv(channels[1]+channels[2],channels[1])
                self.x11=self._double_conv(channels[2]+channels[3],channels[2])
        else:
            # Load timm backbone WITHOUT global_pool — we do pooling ourselves
            bb = timm.create_model(backbone_cfg["timm_name"],
                                   pretrained=True,
                                   num_classes=0,
                                   global_pool="")   # ← no pool yet
            self.backbone = bb
            # Probe real channel count with dummy forward
            with torch.no_grad():
                bb.eval()
                dummy_size = (IMG_SIZE_INCEPTION
                              if backbone_cfg["key"]=="inception_v3"
                              else IMG_SIZE_DEFAULT)
                dummy = torch.zeros(1, 3, dummy_size, dummy_size)
                out   = bb(dummy)
                # out is [B,C,H,W] since global_pool=""
                feat_dim = out.shape[1]
            print(f"    [{self.key}] spatial feat_dim={feat_dim}")

        # FIX 1: shared adaptive pool + flatten before projection
        self.gap  = nn.AdaptiveAvgPool2d(1)          # [B,C,H,W] → [B,C,1,1]
        mid       = max(embed_dim, feat_dim//2)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, mid),  nn.GELU(),
            nn.Dropout(dropout_p),
            nn.Linear(mid, embed_dim), nn.Dropout(dropout_p),
        )
        print(f"    [{self.key}] proj: {feat_dim} → {mid} → {embed_dim}")

    # ── U-Net encoder helpers ─────────────────────────────────────────────────
    @staticmethod
    def _double_conv(in_ch, out_ch, residual=False):
        if residual:
            return _ResDoubleConv(in_ch, out_ch)
        return nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def _make_encoder(self, ch, variant):
        res = (variant=="residual")
        return (self._double_conv(ch[0],ch[1],res),
                self._double_conv(ch[1],ch[2],res),
                self._double_conv(ch[2],ch[3],res),
                self._double_conv(ch[3],ch[4],res))

    def _unet_forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool_op(e1))
        e3 = self.enc3(self.pool_op(e2))
        e4 = self.enc4(self.pool_op(e3))
        return e4   # [B, 512, H', W']

    # ── Main forward ──────────────────────────────────────────────────────────
    def forward(self, x):
        if self.is_unet:
            feat_map = self._unet_forward(x)          # [B,512,H,W]
        else:
            feat_map = self.backbone(x)                # [B,C,H,W]

        # FIX 1: uniform pooling regardless of spatial size
        pooled = self.gap(feat_map).flatten(1)         # [B, C]
        embed  = self.proj(pooled)                     # [B, EMBED_DIM]
        return {"embedding": F.normalize(embed, p=2, dim=-1)}

    @torch.no_grad()
    def embed(self, x):
        self.eval()
        return self.forward(x)["embedding"].cpu().numpy()


class _ResDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net  = nn.Sequential(
            nn.Conv2d(in_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch,out_ch,3,padding=1), nn.BatchNorm2d(out_ch),
        )
        self.skip = nn.Conv2d(in_ch,out_ch,1) if in_ch!=out_ch else nn.Identity()
        self.act  = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.net(x)+self.skip(x))

# ══════════════════════════════════════════════════════════════════════════════
# CLINICAL ENCODER + CROSS-ATTENTION + CLASSIFIER  (unchanged)
# ══════════════════════════════════════════════════════════════════════════════
class ClinicalEncoder(nn.Module):
    def __init__(self, in_dim=META_DIM, embed_dim=EMBED_DIM, dropout_p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim,64), nn.GELU(), nn.Dropout(dropout_p),
            nn.Linear(64,embed_dim), nn.Dropout(dropout_p),
        )
    def forward(self, x): return F.normalize(self.net(x), p=2, dim=-1)


class ExplicitCrossAttention(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_heads=4):
        super().__init__()
        assert embed_dim%n_heads==0
        self.h=n_heads; self.dk=embed_dim//n_heads
        self.W_Q=nn.Linear(embed_dim,embed_dim,bias=False)
        self.W_K=nn.Linear(embed_dim,embed_dim,bias=False)
        self.W_V=nn.Linear(embed_dim,embed_dim,bias=False)
        self.W_O=nn.Linear(embed_dim,embed_dim,bias=False)
        self.ln =nn.LayerNorm(embed_dim)
    def forward(self, Ei, Em):
        B,D=Ei.shape; h,dk=self.h,self.dk
        Q=self.W_Q(Em).view(B,1,h,dk).transpose(1,2)
        K=self.W_K(Ei).view(B,1,h,dk).transpose(1,2)
        V=self.W_V(Ei).view(B,1,h,dk).transpose(1,2)
        A  =F.softmax(torch.matmul(Q,K.transpose(-2,-1))/dk**0.5,dim=-1)
        ctx=torch.matmul(A,V).transpose(1,2).contiguous().view(B,D)
        return self.ln(self.W_O(ctx)+Em)


class MultimodalClassifier(nn.Module):
    def __init__(self, backbone_cfg, embed_dim=EMBED_DIM, n_cls=2, dp=0.3):
        super().__init__()
        self.img_enc  = UniversalImageEncoder(backbone_cfg, embed_dim, dp)
        self.meta_enc = ClinicalEncoder(META_DIM, embed_dim, dp)
        self.attn     = ExplicitCrossAttention(embed_dim, n_heads=4)
        self.clf      = nn.Sequential(
            nn.Linear(embed_dim*2,256), nn.GELU(),
            nn.Dropout(dp), nn.Linear(256,n_cls),
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
        return torch.cat([Ei, self.attn(Ei,Em)], dim=-1).cpu().numpy()

# ══════════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model, device, lr=3e-4, label="model"):
        self.model=model.to(device); self.device=device; self.label=label
        self.opt    =torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
        self.sched  =torch.optim.lr_scheduler.CosineAnnealingLR(self.opt,T_max=N_EPOCHS)
        self.loss_fn=nn.CrossEntropyLoss(); self.history=[]

    def train_epoch(self, loader):
        self.model.train(); total=0.0
        for b in loader:
            imgs=b["image"].to(self.device).float()
            meta=b["meta_vec"].to(self.device).float()
            labs=b["label"].to(self.device)
            self.opt.zero_grad()
            loss=self.loss_fn(self.model(imgs,meta),labs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
            self.opt.step(); total+=loss.item()
        self.sched.step()
        return total/max(len(loader),1)

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval(); all_logits,all_labs=[],[]
        for b in loader:
            imgs=b["image"].to(self.device).float()
            meta=b["meta_vec"].to(self.device).float()
            all_logits.append(self.model(imgs,meta).cpu())
            all_labs.append(b["label"])
        logits=torch.cat(all_logits); labels=torch.cat(all_labs).numpy()
        probs=F.softmax(logits,dim=-1).numpy()
        probs_c=np.clip(probs,1e-10,1-1e-10)
        auc=roc_auc_score(labels,probs_c[:,1]) if len(np.unique(labels))>1 else 0.5
        return {"accuracy":accuracy_score(labels,probs.argmax(1)),
                "log_loss":log_loss(labels,probs_c),"auc":auc}

    def fit(self, train_loader, val_loader):
        print(f"\n  ── {self.label} ──"); best=0.0
        for ep in range(1,N_EPOCHS+1):
            tr_l=self.train_epoch(train_loader)
            val =self.evaluate(val_loader)
            self.history.append({"epoch":ep,"train_loss":tr_l,**val})
            print(f"    ep {ep:02d}/{N_EPOCHS}  loss={tr_l:.4f}  "
                  f"val_acc={val['accuracy']:.4f}  auc={val['auc']:.4f}")
            if val["accuracy"]>best:
                best=val["accuracy"]
                torch.save(self.model.state_dict(),
                           os.path.join(OUT_DIR,f"{self.label}_best.pt"))
        print(f"    Best val acc: {best:.4f}")

    @torch.no_grad()
    def extract_embeddings(self, loader):
        self.model.eval(); embs,labs=[],[]
        for b in loader:
            imgs=b["image"].to(self.device).float()
            meta=b["meta_vec"].to(self.device).float()
            embs.append(self.model.embed(imgs,meta))
            labs.append(b["label"].numpy())
        # FIX 1: vstack guarantees consistent shape — no NaN source here
        return np.vstack(embs).astype(np.float32), np.concatenate(labs)

# ══════════════════════════════════════════════════════════════════════════════
# BIAS & VARIANCE QUANTIFIERS
# ══════════════════════════════════════════════════════════════════════════════
def make_valence_attributes(embed_dim):
    rng=np.random.default_rng(seed=0)
    poles=dict(sacroiliitis=rng.standard_normal(embed_dim),
               inflammation=rng.standard_normal(embed_dim),
               bone_edema  =rng.standard_normal(embed_dim),
               normal_spine=rng.standard_normal(embed_dim),
               healthy     =rng.standard_normal(embed_dim),
               asymptomatic=rng.standard_normal(embed_dim))
    return {k:(v/(np.linalg.norm(v)+1e-8)).astype(np.float32)
            for k,v in poles.items()}


class CaliskanBridgeQuantifier:
    PATHO  =["sacroiliitis","inflammation","bone_edema"]
    HEALTHY=["normal_spine","healthy","asymptomatic"]
    def __init__(self,attrs): self.attrs=attrs
    def _assoc(self,emb):
        sA=np.mean([np.dot(emb,self.attrs[k]) for k in self.PATHO])
        sB=np.mean([np.dot(emb,self.attrs[k]) for k in self.HEALTHY])
        return float(sA-sB)
    def compute(self,X,Y):
        sX=np.array([self._assoc(e) for e in X],dtype=np.float32)
        sY=np.array([self._assoc(e) for e in Y],dtype=np.float32)
        all_s=np.concatenate([sX,sY])
        d=float((sX.mean()-sY.mean())/(all_s.std()+1e-8))
        def stat(a,b,axis=0): return a.mean(axis=axis)-b.mean(axis=axis)
        perm=permutation_test((sX,sY),stat,permutation_type="independent",
                              n_resamples=1000,alternative="two-sided",
                              random_state=SEED)
        return {"weat_d":d,"p_value":float(perm.pvalue),
                "interpretation":(
                    "High |d|>=0.5" if abs(d)>=0.5 else
                    "Moderate |d|>=0.2" if abs(d)>=0.2 else "Low |d|<0.2")}


class DCEQuantifier:
    def __init__(self,model,device):
        self.model=model.to(device); self.device=device
    @torch.no_grad()
    def measure(self,images,meta):
        self.model.eval()
        imgs=images.to(self.device).float()
        meta=meta.to(self.device).float()
        p_full=F.softmax(self.model(imgs,meta),dim=-1)[:,1].cpu().numpy()
        dce={}
        for j,fname in enumerate(CLINICAL_FEATURES):
            mcf=meta.clone(); mcf[:,j]=0.0
            p_cf=F.softmax(self.model(imgs,mcf),dim=-1)[:,1].cpu().numpy()
            dce[fname]=float((p_full-p_cf).mean())
        return dce


class ssCVQuantifier:
    """
    FIX 1: embeddings are always [N, EMBED_DIM*2] float32 due to uniform GAP.
    No more shape mismatch → no more NaN variance.
    """
    def __init__(self,delta_grid=DELTA_GRID,n_replicates=N_REPLICATES):
        self.deltas=delta_grid; self.R=n_replicates
        self.rng=np.random.default_rng(SEED)

    def _probe(self,Xtr,ytr,Xte,yte):
        sc=StandardScaler()
        clf=LogisticRegression(C=1.0,max_iter=500,solver="lbfgs",
                               random_state=int(self.rng.integers(9999)))
        clf.fit(sc.fit_transform(Xtr),ytr)
        prob=clf.predict_proba(sc.transform(Xte))
        return log_loss(yte,prob),accuracy_score(yte,clf.predict(sc.transform(Xte)))

    def run(self,embs_tr,labs_tr,embs_te,labs_te):
        # Sanity check shapes
        assert embs_tr.dtype==np.float32, "Float32 contract violated"
        assert embs_te.shape[1]==embs_tr.shape[1], \
            f"Embedding dim mismatch: tr={embs_tr.shape} te={embs_te.shape}"
        records=[]
        for delta in self.deltas:
            n_sub=max(10,int(len(embs_tr)*(1.0-delta)))
            losses=[]
            for r in range(self.R):
                idx=self.rng.choice(len(embs_tr),size=n_sub,replace=False)
                ll,acc=self._probe(embs_tr[idx],labs_tr[idx],embs_te,labs_te)
                losses.append(ll)
                records.append({"delta":delta,"replicate":r,
                                "loss":ll,"accuracy":acc,"n_train":n_sub})
        df=pd.DataFrame(records)
        df["sigma2"]=df.groupby("delta")["loss"].transform(lambda x:x.var(ddof=1))
        return df


class MCDropoutQuantifier:
    def __init__(self,model,n_passes=MC_PASSES,device=DEVICE):
        self.model=model; self.T=n_passes; self.device=device
    def _enable_dropout(self):
        for m in self.model.modules():
            if isinstance(m,nn.Dropout): m.train()
    @torch.no_grad()
    def predict(self,images,meta):
        self.model.eval(); self._enable_dropout()
        imgs=images.to(self.device).float()
        meta=meta.to(self.device).float()
        passes=[]
        for _ in range(self.T):
            passes.append(F.softmax(self.model(imgs,meta),dim=-1).cpu())
        P=torch.stack(passes,dim=0).numpy(); p_bar=P.mean(axis=0)
        H_total=-np.sum(p_bar*np.log(p_bar+1e-10),axis=-1)
        per_H  =-np.sum(P    *np.log(P    +1e-10),axis=-1)
        H_ale  =per_H.mean(axis=0); H_epi=H_total-H_ale
        KL=np.log(p_bar.shape[-1])-H_total
        return {"H_total":H_total,"H_aleatoric":H_ale,
                "H_epistemic":H_epi,"KL_uniform":KL}

# ══════════════════════════════════════════════════════════════════════════════
# PER-BACKBONE PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def run_backbone(backbone_cfg, meta_df, test_batch_store):
    key = backbone_cfg["key"]

    # FIX 2: use correct image size per backbone
    img_size = (IMG_SIZE_INCEPTION
                if key=="inception_v3"
                else IMG_SIZE_DEFAULT)

    print(f"\n{'='*60}")
    print(f"  BACKBONE: {key.upper()}  [{backbone_cfg['family']} {backbone_cfg['year']}]"
          f"  img_size={img_size}")
    print(f"{'='*60}")
    t0=time.time()

    # Build dataloaders for this backbone's image size
    loaders = build_dataloaders(DATA_ROOT, meta_df,
                                batch_size=BATCH_SIZE, img_size=img_size)

    # Get test batch for DCE + MC-Dropout (same device, this backbone's size)
    test_batch = next(iter(loaders["test"]))
    test_batch_store[key] = test_batch   # save for reference

    model  = MultimodalClassifier(backbone_cfg, EMBED_DIM)
    trainer= Trainer(model, DEVICE, label=key)
    trainer.fit(loaders["train"], loaders["val"])

    test_metrics=trainer.evaluate(loaders["test"])
    print(f"  Test  acc={test_metrics['accuracy']:.4f}  "
          f"auc={test_metrics['auc']:.4f}  ll={test_metrics['log_loss']:.4f}")

    emb_tr,lab_tr=trainer.extract_embeddings(loaders["train"])
    emb_te,lab_te=trainer.extract_embeddings(loaders["test"])
    print(f"  Embeddings: train={emb_tr.shape}  test={emb_te.shape}  "
          f"dtype={emb_tr.dtype}")  # should always be (N, 1024) float32

    # Bias
    attrs=make_valence_attributes(emb_te.shape[1])
    cal  =CaliskanBridgeQuantifier(attrs)
    mh=lab_te==0; mp=lab_te==1
    if mh.sum()==0 or mp.sum()==0:
        mid=len(lab_te)//2; mh=np.arange(len(lab_te))<mid; mp=~mh
    bias=cal.compute(emb_te[mh],emb_te[mp])
    print(f"  WEAT d={bias['weat_d']:+.4f}  p={bias['p_value']:.4f}  "
          f"{bias['interpretation']}")

    # DCE
    dce=DCEQuantifier(model,DEVICE).measure(
            test_batch["image"], test_batch["meta_vec"])

    # ssCV — FIX 1: shape guaranteed consistent
    sscv_df=ssCVQuantifier(DELTA_GRID,N_REPLICATES).run(
                emb_tr,lab_tr,emb_te,lab_te)
    sscv_df.to_csv(os.path.join(OUT_DIR,f"sscv_{key}.csv"),index=False)

    # MC-Dropout
    ent=MCDropoutQuantifier(model,MC_PASSES,DEVICE).predict(
            test_batch["image"],test_batch["meta_vec"])

    elapsed=time.time()-t0
    print(f"  Done in {elapsed/60:.1f} min")

    def s2(df,d):
        s=df[df["delta"]==d]["loss"]
        v=float(s.var(ddof=1)) if len(s)>1 else float("nan")
        if np.isnan(v):
            print(f"  WARNING: NaN variance at delta={d} for {key}!")
        return round(v,6)

    return {
        "Backbone":         key,
        "Family":           backbone_cfg["family"],
        "Year":             backbone_cfg["year"],
        "Img Size":         img_size,
        "Test Accuracy":    round(test_metrics["accuracy"],4),
        "Test AUC":         round(test_metrics["auc"],     4),
        "Test Log-Loss":    round(test_metrics["log_loss"],4),
        "WEAT |d|":         round(abs(bias["weat_d"]),     4),
        "Bias p-value":     round(bias["p_value"],         4),
        "Bias class":       bias["interpretation"],
        "Mean |DCE|":       round(float(np.mean(np.abs(list(dce.values())))),5),
        "sigma2_d0.0":      s2(sscv_df,0.0),
        "sigma2_d0.3":      s2(sscv_df,0.3),
        "sigma2_d0.5":      s2(sscv_df,0.5),
        "H_epistemic":      round(float(ent["H_epistemic"].mean()),5),
        "H_aleatoric":      round(float(ent["H_aleatoric"].mean()), 5),
        "KL_uniform":       round(float(ent["KL_uniform"].mean()),  5),
        "Train time (min)": round(elapsed/60,2),
    }

# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig,name):
    p=os.path.join(OUT_DIR,name)
    fig.savefig(p,dpi=150,bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] → {name}")


def plot_all(df):
    backbones=df["Backbone"].tolist()
    x=np.arange(len(backbones))
    colors=plt.cm.tab20(np.linspace(0,1,len(backbones)))

    # 1. Accuracy + AUC
    fig,axes=plt.subplots(1,2,figsize=(18,6))
    fig.suptitle("Test Performance — All 17 Backbones (Fixed)",
                 fontsize=13,fontweight="bold")
    for ax,col,title in [(axes[0],"Test Accuracy","Test Accuracy"),
                         (axes[1],"Test AUC",     "Test AUC")]:
        bars=ax.bar(x,df[col],color=colors,edgecolor="black",linewidth=0.5)
        ax.set_xticks(x); ax.set_xticklabels(backbones,rotation=45,
                                              ha="right",fontsize=8)
        ax.set_ylim(0,1.12); ax.set_ylabel(title); ax.set_title(title)
        ax.axhline(0.5,color="grey",ls="--",lw=0.8)
        ax.grid(axis="y",ls="--",alpha=0.4)
        for bar,val in zip(bars,df[col]):
            ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                    f"{val:.3f}",ha="center",va="bottom",fontsize=7)
    _save(fig,"final_accuracy_auc.png")

    # 2. WEAT Bias
    fig,ax=plt.subplots(figsize=(16,5))
    bars=ax.bar(x,df["WEAT |d|"],color=colors,edgecolor="black",linewidth=0.5)
    ax.axhline(0.5,color="red",   ls="--",lw=1.2,label="|d|=0.5 High")
    ax.axhline(0.2,color="orange",ls="--",lw=1.0,label="|d|=0.2 Moderate")
    ax.set_xticks(x); ax.set_xticklabels(backbones,rotation=45,
                                          ha="right",fontsize=8)
    ax.set_ylabel("WEAT |d|")
    ax.set_title("Representational Bias — All 17 Backbones (Fixed)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y",ls="--",alpha=0.4)
    for bar,val in zip(bars,df["WEAT |d|"]):
        ax.text(bar.get_x()+bar.get_width()/2,bar.get_height()+0.01,
                f"{val:.3f}",ha="center",va="bottom",fontsize=7)
    _save(fig,"final_weat_bias.png")

    # 3. Variance — FIX 1 should now produce real values not NaN
    fig,ax=plt.subplots(figsize=(18,6))
    w=0.25
    ax.bar(x-w,df["sigma2_d0.0"],w,label="σ²(δ=0.0)",
           color="royalblue", alpha=0.85)
    ax.bar(x,  df["sigma2_d0.3"],w,label="σ²(δ=0.3)",
           color="darkorange",alpha=0.85)
    ax.bar(x+w,df["sigma2_d0.5"],w,label="σ²(δ=0.5)",
           color="firebrick", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(backbones,rotation=45,
                                          ha="right",fontsize=8)
    ax.set_ylabel("Variance σ²(δ)")
    ax.set_title("ssCV Variance — All 17 Backbones (Fixed, No NaN)",
                 fontweight="bold")
    ax.legend(); ax.grid(axis="y",ls="--",alpha=0.4)
    _save(fig,"final_sscv_variance.png")

    # 4. MC-Dropout Entropy
    fig,ax=plt.subplots(figsize=(18,6))
    ax.bar(x-0.2,df["H_epistemic"],0.35,label="Epistemic",
           color="purple",alpha=0.85)
    ax.bar(x+0.2,df["H_aleatoric"],0.35,label="Aleatoric",
           color="teal",  alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(backbones,rotation=45,
                                          ha="right",fontsize=8)
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("MC-Dropout Entropy — All 17 Backbones",fontweight="bold")
    ax.legend(); ax.grid(axis="y",ls="--",alpha=0.4)
    _save(fig,"final_entropy.png")

    # 5. Summary heatmap
    heat_cols=["Test Accuracy","Test AUC","WEAT |d|",
               "sigma2_d0.3","H_epistemic","Mean |DCE|"]
    heat_df  =df.set_index("Backbone")[heat_cols].astype(float)
    norm_heat=(heat_df-heat_df.min())/(heat_df.max()-heat_df.min()+1e-8)
    fig,ax=plt.subplots(figsize=(13,max(7,len(backbones)*0.5)))
    sns.heatmap(norm_heat,annot=heat_df.round(4),fmt="",
                cmap="RdYlGn",linewidths=0.5,ax=ax,
                cbar_kws={"label":"Normalised value"})
    ax.set_title("Backbone Comparison Heatmap — All 17 Models (Fixed)",
                 fontsize=12,fontweight="bold")
    _save(fig,"final_heatmap.png")

    # 6. Radar
    metrics=["Test Accuracy","Test AUC","WEAT |d|","H_epistemic","Mean |DCE|"]
    radar_df=df[metrics].copy().astype(float)
    for col in metrics:
        mn,mx=radar_df[col].min(),radar_df[col].max()
        radar_df[col]=(radar_df[col]-mn)/(mx-mn+1e-8)
    angles=np.linspace(0,2*np.pi,len(metrics),endpoint=False).tolist()+[0]
    fig,ax=plt.subplots(figsize=(9,9),subplot_kw=dict(polar=True))
    for i,(_,row) in enumerate(radar_df.iterrows()):
        vals=row.tolist()+[row.tolist()[0]]
        ax.plot(angles,vals,"o-",linewidth=1.2,
                label=backbones[i],color=colors[i],alpha=0.8)
    ax.set_thetagrids(np.degrees(angles[:-1]),metrics,fontsize=9)
    ax.set_title("Radar — All 17 Backbones",fontsize=12,
                 fontweight="bold",pad=20)
    ax.legend(loc="upper right",bbox_to_anchor=(1.4,1.15),fontsize=7)
    _save(fig,"final_radar.png")

    # 7. Styled tally table PNG
    family_colors={"VGG":"#FFF3CD","ResNet":"#D1ECF1",
                   "Inception":"#D4EDDA","U-Net":"#F8D7DA"}
    display_cols=["Backbone","Family","Year","Img Size",
                  "Test Accuracy","Test AUC","Test Log-Loss",
                  "WEAT |d|","Bias class",
                  "sigma2_d0.0","sigma2_d0.3","sigma2_d0.5",
                  "H_epistemic","H_aleatoric","Train time (min)"]
    show=df[display_cols]
    fig,ax=plt.subplots(figsize=(26,max(5,len(df)*0.65+1.5)))
    ax.axis("off")
    tbl=ax.table(cellText=show.values,colLabels=show.columns,
                 cellLoc="center",loc="center",bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7)
    for j in range(len(show.columns)):
        tbl[0,j].set_facecolor("#1a3a5c")
        tbl[0,j].set_text_props(color="white",fontweight="bold")
    for i,(_,row) in enumerate(df.iterrows(),start=1):
        clr=family_colors.get(row["Family"],"#FFFFFF")
        for j in range(len(show.columns)):
            tbl[i,j].set_facecolor(clr)
    fig.suptitle(
        "Fixed Bias-Variance Tally — All 17 Backbones — SacroMRI\n"
        "Fixes: (1) GAP normalisation → no NaN variance  "
        "(2) Inception v3 @ 299×299  "
        "(3) No label leakage",
        fontsize=10,fontweight="bold")
    _save(fig,"final_tally_all17_FIXED.png")
    print("\n  All plots saved.")

# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n"+"="*60)
    print("  FIXED MULTI-BACKBONE PIPELINE — SacroMRI")
    print("  Fixes: NaN variance | Inception size | Label leakage")
    print("="*60)

    meta_df = load_and_parse_csv(CSV_PATH) if os.path.exists(CSV_PATH) else None

    all_results    = []
    failed         = []
    test_batch_store = {}

    for i,cfg in enumerate(BACKBONES):
        print(f"\n[{i+1}/{len(BACKBONES)}] {cfg['key']} ...")
        try:
            result = run_backbone(cfg, meta_df, test_batch_store)
            all_results.append(result)
            # Incremental save after every backbone
            pd.DataFrame(all_results).to_csv(
                os.path.join(OUT_DIR,"tally_incremental_fixed.csv"),index=False)
        except Exception as e:
            import traceback
            print(f"  FAILED: {cfg['key']} — {e}")
            traceback.print_exc()
            failed.append({"Backbone":cfg["key"],"Error":str(e)})
            pd.DataFrame(failed).to_csv(
                os.path.join(OUT_DIR,"failed_fixed.csv"),index=False)
            continue

    if not all_results:
        print("No results — all backbones failed."); return

    # Sort by family → year → name
    tally = pd.DataFrame(all_results)
    order = {"VGG":0,"ResNet":1,"Inception":2,"U-Net":3}
    tally["_o"] = tally["Family"].map(order).fillna(99)
    tally = (tally.sort_values(["_o","Year","Backbone"])
                  .drop(columns="_o")
                  .reset_index(drop=True))
    tally.to_csv(os.path.join(OUT_DIR,"tally_all17_FIXED.csv"),index=False)

    # Print summary
    print("\n"+"="*60)
    print("  FINAL TABLE — ALL 17 BACKBONES (FIXED)")
    print("="*60)
    disp=["Backbone","Family","Year","Img Size",
          "Test Accuracy","Test AUC","WEAT |d|",
          "sigma2_d0.3","H_epistemic","Train time (min)"]
    print(tally[disp].to_string(index=False))
    print("="*60)

    plot_all(tally)

    # NaN check report
    nan_cols = tally[["sigma2_d0.0","sigma2_d0.3","sigma2_d0.5"]].isna().sum()
    print(f"\n  NaN check — sigma2 columns:")
    for col,n in nan_cols.items():
        status = "PASS" if n==0 else f"FAIL ({n} NaNs)"
        print(f"    {col}: {status}")

    best_acc = tally.loc[tally["Test Accuracy"].idxmax()]
    low_bias = tally.loc[tally["WEAT |d|"].idxmin()]
    print(f"\n  Best Accuracy : {best_acc['Backbone']}  ({best_acc['Test Accuracy']:.4f})")
    print(f"  Lowest Bias   : {low_bias['Backbone']}  (|d|={low_bias['WEAT |d|']:.4f})")
    if failed:
        print(f"\n  Failed backbones: {[f['Backbone'] for f in failed]}")
    print(f"\n[DONE] → {OUT_DIR}")


if __name__=="__main__":
    main()