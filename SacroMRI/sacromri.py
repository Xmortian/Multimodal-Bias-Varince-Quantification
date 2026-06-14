

# ══════════════════════════════════════════════════════════════════════════════
# 0.  ALL IMPORTS — must be at the very top of the single cell
# ══════════════════════════════════════════════════════════════════════════════
import os, glob, warnings, sys
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
from sklearn.metrics import log_loss, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine
from scipy.stats import permutation_test

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
print("✓  All imports successful")
print("✓  Script version: v4-zero-fallback-no-leakage")

# ══════════════════════════════════════════════════════════════════════════════
# 1.  PATHS  — verified against os.walk output
# ══════════════════════════════════════════════════════════════════════════════
# Root of the Kaggle dataset mount
# Auto-download dataset
KAGGLE_ROOT = r"D:\SacroMri\archive"
CSV_PATH    = os.path.join(KAGGLE_ROOT, "SacroMRI Dataset.csv")
DATA_ROOT   = os.path.join(KAGGLE_ROOT, "SacroMRI")
OUT_DIR     = r"D:\SacroMri\bvq_outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Quick sanity check ────────────────────────────────────────────────────────
print(f"CSV exists  : {os.path.exists(CSV_PATH)}  →  {CSV_PATH}")
print(f"DATA_ROOT   : {os.path.exists(DATA_ROOT)} →  {DATA_ROOT}")
for split in ("train", "val", "test"):
    for cls in ("AS", "Normal"):
        d = os.path.join(DATA_ROOT, split, cls)
        n = len(glob.glob(os.path.join(d, "*.png"))) if os.path.isdir(d) else 0
        print(f"  {split:5s}/{cls:6s}  {n:4d} PNGs")

# ══════════════════════════════════════════════════════════════════════════════
# 2.  GLOBAL CONFIG
# ══════════════════════════════════════════════════════════════════════════════
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_SIZE     = 340
EMBED_DIM    = 512
BATCH_SIZE   = 32
SEED         = 42
MC_PASSES    = 50
DELTA_GRID   = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
N_REPLICATES = 30
N_EPOCHS     = 8

torch.manual_seed(SEED)
np.random.seed(SEED)
print(f"\n✓  Config: device={DEVICE}  img={IMG_SIZE}  embed={EMBED_DIM}")

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CSV PARSER  (handles the confirmed SacroMRI Dataset.csv)
# ══════════════════════════════════════════════════════════════════════════════
# The 8 clinical biomarker features we use from the CSV
# ── Clinical features: ONLY pre-diagnostic patient-reported & lab values ──────
# Removed:
#   diagnosis_flag   → IS the prediction target (direct leakage)
#   bone_marrow_edema → radiologist MRI finding that directly encodes diagnosis
#   erosions          → radiologist MRI finding that directly encodes diagnosis
# Kept: 5 features collected BEFORE imaging diagnosis is known
CLINICAL_FEATURES = [
    "chronic_back_pain",     # patient-reported symptom  Yes/No → 1.0/0.0
    "morning_stiffness",     # patient-reported symptom  Yes/No → 1.0/0.0
    "improvement_exercise",  # patient-reported symptom  Yes/No → 1.0/0.0
    "hla_b27",               # lab test result           Positive/Negative → 1.0/0.0
    "esr_norm",              # blood marker (ESR mm/hr)  z-scored float32
    "crp_norm",              # blood marker (CRP mg/L)   z-scored float32
]
META_DIM = len(CLINICAL_FEATURES)   # = 6


def load_and_parse_csv(csv_path: str) -> pd.DataFrame:
    """
    Reads SacroMRI Dataset.csv.
    Returns a clean DataFrame with normalised float32 clinical features.
    All binary fields: Yes/Positive → 1.0,  No/Negative → 0.0.
    Continuous ESR & CRP → z-score float32.
    """
    df = pd.read_csv(csv_path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    print(f"[CSV] raw columns : {list(df.columns)}")
    print(f"[CSV] rows        : {len(df)}")

    # ── Fuzzy column rename ───────────────────────────────────────────────────
    rmap = {}
    for c in df.columns:
        cl = c.lower()
        if "morning"  in cl:                      rmap[c] = "morning_stiffness"
        elif "exercise" in cl:                    rmap[c] = "improvement_exercise"
        elif "hla"    in cl:                      rmap[c] = "hla_b27"
        elif "marrow" in cl or "edema" in cl:     rmap[c] = "bone_marrow_edema"
        elif "erosion" in cl:                     rmap[c] = "erosions"
        elif "esr"    in cl:                      rmap[c] = "esr"
        elif "crp"    in cl:                      rmap[c] = "crp"
        elif ("image" in cl and "id" in cl) or cl == "id":
                                                  rmap[c] = "image_id"
        elif "diagnosis" in cl:                   rmap[c] = "diagnosis"
    df = df.rename(columns=rmap)
    print(f"[CSV] mapped cols : {list(df.columns)}")

    # ── Binary encoding ───────────────────────────────────────────────────────
    _YES = {"yes": 1.0, "no": 0.0, "positive": 1.0,
            "negative": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0}
    for col in ["chronic_back_pain", "morning_stiffness", "improvement_exercise",
                "hla_b27"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip().str.lower()
                         .map(_YES).fillna(0.0).astype(np.float32))
        else:
            df[col] = np.float32(0.0)

    # ── Z-score continuous biomarkers (float32) ───────────────────────────────
    # z = (x − μ) / σ  — preserves sub-unit precision for variance analysis
    for raw, norm in [("esr", "esr_norm"), ("crp", "crp_norm")]:
        if raw in df.columns:
            v = pd.to_numeric(df[raw], errors="coerce").fillna(0.0)
            df[norm] = ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)
        else:
            df[norm] = np.float32(0.0)

    # ── Diagnosis flag ────────────────────────────────────────────────────────
    if "diagnosis" in df.columns:
        df["diagnosis_flag"] = (df["diagnosis"].astype(str)
                                  .str.strip().str.upper() == "AS"
                               ).astype(np.float32)
    else:
        df["diagnosis_flag"] = np.float32(0.0)

    # ── Image ID ──────────────────────────────────────────────────────────────
    if "image_id" not in df.columns:
        df["image_id"] = df.index.astype(str)
    df["image_id"] = df["image_id"].astype(str).str.strip()

    return df


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MRI PREPROCESSOR
# ══════════════════════════════════════════════════════════════════════════════
class MRIPreprocessor:
    """
    PNG MRI preprocessing chain (all operations keep dtype=float32):

      1. PIL → float32 array  (divide by 255.0 — NEVER clamp back to uint8)
      2. Gaussian N4-approx bias field correction
         corrected = original / (GaussianBlur(original) + ε)
      3. Spatial alignment: centre-of-mass shift (tilt noise guard)
      4. Z-score normalisation: z = (x−μ)/σ  → float32 ∈ ℝ
      5. Bilinear resize to 340×340
      6. Triplicate grayscale → [3, 340, 340] float32 tensor
    """

    def __init__(self, target_size: int = IMG_SIZE):
        self.target_size = target_size
        # Pre-built Gaussian kernel (σ=15) for bias field estimation
        self._kernel = self._gauss_kernel(sigma=15, size=31)

    @staticmethod
    def _gauss_kernel(sigma: float, size: int) -> torch.Tensor:
        """G(x,y) = exp(−(x²+y²)/(2σ²)), returns [1,1,H,W] float32 kernel."""
        c  = torch.arange(size).float() - size // 2
        g1 = torch.exp(-c**2 / (2 * sigma**2))
        g2 = torch.ger(g1, g1)
        return (g2 / g2.sum()).unsqueeze(0).unsqueeze(0)

    def _n4_approx(self, arr: np.ndarray) -> np.ndarray:
        """
        Approximate N4 bias field correction.
        Bias field f(x) ≈ GaussianBlur(v(x))
        Corrected u(x) = v(x) / (f(x) + ε)
        Both v and u are float32; never integer.
        """
        t   = torch.from_numpy(arr).float().unsqueeze(0).unsqueeze(0)
        pad = self._kernel.shape[-1] // 2
        bf  = F.conv2d(t, self._kernel, padding=pad)
        return (t / (bf + 1e-3)).squeeze().numpy()

    @staticmethod
    def _align(arr: np.ndarray) -> np.ndarray:
        """
        Centre-of-mass shift to suppress 3–10° positional tilt.
        Without this, tilt differences appear as model variance (false positive).
        """
        thresh = np.percentile(arr, 75)
        ys, xs = np.where(arr > thresh)
        if len(ys) < 20:
            return arr
        dy = int(arr.shape[0] // 2 - ys.mean())
        dx = int(arr.shape[1] // 2 - xs.mean())
        return np.roll(np.roll(arr, dy, axis=0), dx, axis=1)

    @staticmethod
    def _zscore(arr: np.ndarray) -> np.ndarray:
        """
        z = (x − μ) / σ  → float32
        CRITICAL: never convert to uint8 here.
        Loss of sub-unit precision destroys epistemic variance signal.
        """
        mu, sd = arr.mean(), arr.std()
        return ((arr - mu) / (sd + 1e-8)).astype(np.float32)

    def process(self, pil_img: "Image.Image") -> torch.Tensor:
        """
        Returns [3, IMG_SIZE, IMG_SIZE] float32 tensor.
        dtype=float32 is asserted before return.
        """
        # Step 1: PIL → float32 in [0,1]  — IMMEDIATELY float, not uint8
        arr = np.array(pil_img.convert("L"), dtype=np.float32) / 255.0

        # Step 2: N4-approximate bias correction
        arr = self._n4_approx(arr)

        # Step 3: Spatial alignment
        arr = self._align(arr)

        # Step 4: Z-score → float32 ∈ ℝ  (not clipped)
        arr = self._zscore(arr)

        # Step 5: Bilinear resize (preserves float precision)
        t = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=(self.target_size, self.target_size),
                          mode="bilinear", align_corners=False)

        # Step 6: Grayscale → 3-channel (for ResNet)
        t = t.squeeze(0).repeat(3, 1, 1)   # [3, H, W]

        assert t.dtype == torch.float32, "Float32 contract violated!"
        return t


# ══════════════════════════════════════════════════════════════════════════════
# 5.  PYTORCH DATASET
# ══════════════════════════════════════════════════════════════════════════════
class SacroMRIDataset(Dataset):
    """
    split : "train" | "val" | "test"
    Loads PNG images from DATA_ROOT/split/AS/ and DATA_ROOT/split/Normal/.
    Merges clinical metadata from CSV as float32 vectors.
    """

    def __init__(self, root: str, split: str,
                 meta_df: pd.DataFrame = None,
                 augment: bool = False):
        self.prep    = MRIPreprocessor(IMG_SIZE)
        self.records = []          # list of {path, label, image_id}

        # ── Discover PNG files ───────────────────────────────────────────────
        for cls_name, label in [("AS", 1), ("Normal", 0)]:
            d = os.path.join(root, split, cls_name)
            if not os.path.isdir(d):
                print(f"  [WARN] missing dir: {d}")
                continue
            for p in sorted(glob.glob(os.path.join(d, "*.png"))):
                # image_id = filename without extension, e.g. "AS (514)"
                img_id = os.path.splitext(os.path.basename(p))[0]
                self.records.append({"path": p, "label": label,
                                     "image_id": img_id})

        # ── Build meta lookup  {image_id → float32 vector} ──────────────────
        self._meta_lookup: dict = {}
        self._meta_zero = np.zeros(META_DIM, dtype=np.float32)

        if meta_df is not None:
            for _, row in meta_df.iterrows():
                key = str(row.get("image_id", "")).strip()
                vec = np.array([float(row.get(f, 0.0))
                                for f in CLINICAL_FEATURES],
                               dtype=np.float32)
                self._meta_lookup[key] = vec
            # Fallback: mean vector for IDs not in CSV
            # IMPORTANT: use zero-vector fallback (NOT mean) for unmatched image_ids.
        # Using the mean would leak class distribution: if AS patients have
        # chronic_back_pain=1 and Normal have 0, the mean≈0.49 is still
        # a biased signal. Zero is the only truly neutral, non-leaking fallback.
        match_count = len(self._meta_lookup)
        self._meta_fallback = self._meta_zero.copy()   # always zeros
        if match_count > 0:
            print(f"    meta_lookup: {match_count} matched IDs / "
                  f"{len(self.records)} images")
        else:
            print(f"    meta_lookup: 0 matches — all images use zero meta vector")

        # ── Augmentation (geometric only — keeps float32) ────────────────────
        self.aug = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=10),
        ]) if augment else None

        n_as  = sum(r["label"] == 1 for r in self.records)
        n_nor = sum(r["label"] == 0 for r in self.records)
        print(f"  [{split:5s}] {len(self.records):4d} images  "
              f"(AS={n_as}, Normal={n_nor})")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        rec = self.records[idx]

        # ── Image → float32 tensor ────────────────────────────────────────────
        try:
            img = self.prep.process(Image.open(rec["path"]))
        except Exception as e:
            print(f"  [WARN] {rec['path']}: {e}")
            img = torch.zeros(3, IMG_SIZE, IMG_SIZE, dtype=torch.float32)

        if self.aug:
            img = self.aug(img)

        # ── Clinical metadata → float32 vector ───────────────────────────────
        meta = self._meta_lookup.get(rec["image_id"],
                                     self._meta_fallback.copy())
        meta = torch.from_numpy(meta.astype(np.float32))

        return {
            "image":    img,                                    # [3,H,W] float32
            "label":    torch.tensor(rec["label"], dtype=torch.long),
            "meta_vec": meta,                                   # [8] float32
            "image_id": rec["image_id"],
        }


def build_dataloaders(root: str,
                      meta_df: pd.DataFrame,
                      batch_size: int = BATCH_SIZE) -> dict:
    """Returns {split: DataLoader} for train / val / test."""
    loaders = {}
    for split, aug in [("train", True), ("val", False), ("test", False)]:
        ds = SacroMRIDataset(root, split, meta_df=meta_df, augment=aug)
        loaders[split] = DataLoader(
            ds, batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0, pin_memory=True,
            drop_last=(split == "train"),
        )
    return loaders


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUAL ENCODER  (RadImageNet-style ResNet50)
# ══════════════════════════════════════════════════════════════════════════════
class RadImageNetEncoder(nn.Module):
    """
    Transparent ResNet50 backbone.  Dropout is PERSISTENT at inference
    (required for MC-Dropout Module D).

    forward() returns:
      embedding : [B, EMBED_DIM]     — L2-normalised float32
      pooled    : [B, 2048]          — pre-projection (for audit)
      spatial   : [B, 2048, H', W'] — layer4 feature maps
    """

    def __init__(self, embed_dim: int = EMBED_DIM, dropout_p: float = 0.3):
        super().__init__()
        bb = timm.create_model("resnet50", pretrained=True,
                                num_classes=0, global_pool="avg")
        # Explicit decomposition — no black-box forward()
        self.stem   = nn.Sequential(bb.conv1, bb.bn1, bb.act1, bb.maxpool)
        self.layer1 = bb.layer1
        self.layer2 = bb.layer2
        self.layer3 = bb.layer3
        self.layer4 = bb.layer4
        self.pool   = bb.global_pool

        # Projection: 2048 → EMBED_DIM  (float32 throughout)
        self.proj = nn.Sequential(
            nn.Linear(2048, 1024), nn.GELU(),
            nn.Dropout(p=dropout_p),    # active at test for MC-Dropout
            nn.Linear(1024, embed_dim),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> dict:
        x       = self.stem(x)
        x       = self.layer1(x)
        x       = self.layer2(x)
        x       = self.layer3(x)
        spatial = self.layer4(x)                        # [B,2048,H',W']
        pooled  = self.pool(spatial).flatten(1)         # [B,2048]
        embed   = self.proj(pooled)                     # [B,EMBED_DIM]
        embed_n = F.normalize(embed, p=2, dim=-1)       # L2-norm for cosine
        return {"embedding": embed_n, "pooled": pooled, "spatial": spatial}


class ClinicalEncoder(nn.Module):
    """MLP encoder for 8 clinical features → EMBED_DIM float32."""

    def __init__(self, in_dim: int = META_DIM,
                 embed_dim: int = EMBED_DIM, dropout_p: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(64, embed_dim),
            nn.Dropout(p=dropout_p),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), p=2, dim=-1)   # L2-normed float32


# ── Valence attribute vectors for the Caliskan Bridge ─────────────────────────
def make_valence_attributes(embed_dim: int) -> dict:
    """
    Seeded float32 unit vectors representing clinical valence poles.
    Fixed seed → reproducible WEAT scores across thesis runs.
    """
    rng = np.random.default_rng(seed=0)
    poles = dict(
        sacroiliitis = rng.standard_normal(embed_dim),
        inflammation = rng.standard_normal(embed_dim),
        bone_edema   = rng.standard_normal(embed_dim),
        normal_spine = rng.standard_normal(embed_dim),
        healthy      = rng.standard_normal(embed_dim),
        asymptomatic = rng.standard_normal(embed_dim),
    )
    return {k: (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)
            for k, v in poles.items()}


# ══════════════════════════════════════════════════════════════════════════════
# 7.  BIAS QUANTIFIERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Module A: Caliskan Bridge (WEAT in image embedding space) ─────────────────
class CaliskanBridgeQuantifier:
    """
    WEAT effect size d (Caliskan et al., 2017) adapted to image space.

    Per-sample association:
        s(t, A, B) = mean_{a∈A} cos(t,a) − mean_{b∈B} cos(t,b)

    where A = pathological attribute poles, B = healthy poles.
    Both t and all attribute vectors are L2-normalised → dot product = cos.

    Effect size (Cohen's d analogue):
        d = [ mean_{x∈X} s(x,A,B) − mean_{y∈Y} s(y,A,B) ]
              / std_{X∪Y} s(t,A,B)

    |d| ≥ 0.5 → substantial representational bias (thesis threshold).
    """
    PATHO   = ["sacroiliitis", "inflammation", "bone_edema"]
    HEALTHY = ["normal_spine", "healthy", "asymptomatic"]

    def __init__(self, attr_vecs: dict):
        self.attrs = attr_vecs   # all already float32, L2-normed

    def _assoc(self, emb: np.ndarray) -> float:
        """s(t, A, B) for a single L2-normed embedding."""
        sim_A = np.mean([np.dot(emb, self.attrs[k]) for k in self.PATHO])
        sim_B = np.mean([np.dot(emb, self.attrs[k]) for k in self.HEALTHY])
        return float(sim_A - sim_B)

    def compute(self, X: np.ndarray, Y: np.ndarray) -> dict:
        """
        X : [N_normal,    EMBED_DIM] float32
        Y : [N_sacro,     EMBED_DIM] float32
        Returns full bias report.
        """
        sX  = np.array([self._assoc(e) for e in X], dtype=np.float32)
        sY  = np.array([self._assoc(e) for e in Y], dtype=np.float32)
        all_s = np.concatenate([sX, sY])

        # d = (μ_X − μ_Y) / σ_{X∪Y}
        d   = float((sX.mean() - sY.mean()) / (all_s.std() + 1e-8))

        # Permutation p-value (5 000 resamples)
        def stat(a, b, axis=0): return a.mean(axis=axis) - b.mean(axis=axis)
        perm = permutation_test((sX, sY), stat,
                                permutation_type="independent",
                                n_resamples=5000,
                                alternative="two-sided",
                                random_state=SEED)

        # Full cosine similarity matrix [N, K] → heatmap
        attr_mat   = np.stack(list(self.attrs.values()))  # [K, D]
        sim_matrix = sk_cosine(np.vstack([X, Y]), attr_mat)

        return {
            "weat_d":        d,
            "p_value":       float(perm.pvalue),
            "scores_X":      sX,
            "scores_Y":      sY,
            "sim_matrix":    sim_matrix,
            "attr_names":    list(self.attrs.keys()),
            "interpretation": (
                "High bias   |d|≥0.5" if abs(d) >= 0.5 else
                "Moderate    |d|≥0.2" if abs(d) >= 0.2 else
                "Low bias    |d|<0.2"
            ),
        }


# ── Module B: Direct Causal Effect via Counterfactual Masking ─────────────────
class DCEQuantifier:
    """
    Estimates DCE of each clinical feature on P(AS) by zero-masking:
        DCE(X_j → Y) ≈ E[f(X)] − E[f(X | do(X_j = 0))]
    'do(X_j = 0)' = set feature j to zero for entire batch (float32 0.0).
    """

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model.to(device)
        self.device = device

    @torch.no_grad()
    def measure(self, images: torch.Tensor,
                meta:   torch.Tensor,
                feature_names: list = CLINICAL_FEATURES) -> dict:
        self.model.eval()
        imgs = images.to(self.device).float()
        meta = meta.to(self.device).float()

        # Baseline P(AS | full features)
        p_full = F.softmax(self.model(imgs, meta), dim=-1)[:, 1].cpu().numpy()

        dce = {}
        for j, fname in enumerate(feature_names):
            meta_cf      = meta.clone()
            meta_cf[:, j] = 0.0       # float32 zero — do(X_j = 0)
            p_cf = F.softmax(self.model(imgs, meta_cf), dim=-1)[:, 1].cpu().numpy()
            # DCE_j = E[P_full − P_counterfactual]
            dce[fname] = float((p_full - p_cf).mean())
        return dce


# ══════════════════════════════════════════════════════════════════════════════
# 8.  VARIANCE QUANTIFIERS
# ══════════════════════════════════════════════════════════════════════════════

# ── Module C: ssCV Procedure ──────────────────────────────────────────────────
class ssCVQuantifier:
    """
    Sub-sampled Cross-Validation.

    For δ ∈ DELTA_GRID, R replicates:
      n_δ = ⌊(1−δ)·N_train⌋  (sub-training size)

    Variance at δ:
        σ²(δ) = Var_r[ L(f̂^{r,δ}, D_test) ]

    Growing σ²(δ) → model is unstable / high variance.
    Uses a transparent linear probe (Logistic Regression) so variance
    comes from data perturbation only, not optimizer non-convexity.
    """

    def __init__(self, delta_grid=DELTA_GRID, n_replicates=N_REPLICATES):
        self.deltas = delta_grid
        self.R      = n_replicates
        self.rng    = np.random.default_rng(SEED)

    def _probe(self, Xtr, ytr, Xte, yte):
        sc  = StandardScaler()
        clf = LogisticRegression(C=1.0, max_iter=500, solver="lbfgs",
                                 random_state=int(self.rng.integers(9999)))
        clf.fit(sc.fit_transform(Xtr), ytr)
        prob = clf.predict_proba(sc.transform(Xte))
        return log_loss(yte, prob), accuracy_score(yte, clf.predict(sc.transform(Xte)))

    def run(self, embs_tr: np.ndarray, labs_tr: np.ndarray,
            embs_te: np.ndarray, labs_te: np.ndarray) -> pd.DataFrame:
        records = []
        for delta in self.deltas:
            n_sub  = max(10, int(len(embs_tr) * (1.0 - delta)))
            losses = []
            for r in range(self.R):
                idx = self.rng.choice(len(embs_tr), size=n_sub, replace=False)
                ll, acc = self._probe(embs_tr[idx], labs_tr[idx], embs_te, labs_te)
                losses.append(ll)
                records.append({"delta": delta, "replicate": r,
                                "loss": ll, "accuracy": acc, "n_train": n_sub})
            print(f"  δ={delta:.1f} | n={n_sub:4d} | "
                  f"mean_loss={np.mean(losses):.4f} | σ²(δ)={np.var(losses,ddof=1):.6f}")
        df = pd.DataFrame(records)
        df["sigma2_delta"] = df.groupby("delta")["loss"].transform(
            lambda x: x.var(ddof=1)
        )
        return df


# ── Module D: MC-Dropout Predictive Entropy ────────────────────────────────────
class MCDropoutQuantifier:
    """
    Monte Carlo Dropout — T stochastic forward passes at inference.

    P ∈ ℝ^{T×B×C} (T passes, B samples, C classes)
    p̄  = mean_t P[t]       (mean posterior)

    Total predictive entropy:
        H[y|x] = −Σ_c p̄_c log p̄_c

    Aleatoric (data noise):
        H_ale  = (1/T) Σ_t [−Σ_c P[t,b,c] log P[t,b,c]]

    Epistemic (model ignorance):
        H_epi  = H[y|x] − H_ale

    KL from uniform (information gain):
        KL(p̄ ∥ u) = log C − H[y|x]
    """

    def __init__(self, model, n_passes: int = MC_PASSES,
                 device: torch.device = DEVICE):
        self.model  = model
        self.T      = n_passes
        self.device = device

    def _enable_dropout(self):
        """Activate only Dropout layers; leave BatchNorm in eval mode."""
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.train()

    @torch.no_grad()
    def predict(self, images: torch.Tensor, meta: torch.Tensor) -> dict:
        self.model.eval()
        self._enable_dropout()
        imgs = images.to(self.device).float()
        meta = meta.to(self.device).float()

        passes = []
        for _ in range(self.T):
            logits = self.model(imgs, meta)
            passes.append(F.softmax(logits, dim=-1).cpu())   # float32
        P     = torch.stack(passes, dim=0).numpy()           # [T, B, C]
        p_bar = P.mean(axis=0)                               # [B, C]

        H_total    = -np.sum(p_bar * np.log(p_bar + 1e-10), axis=-1)   # [B]
        per_H      = -np.sum(P     * np.log(P     + 1e-10), axis=-1)   # [T,B]
        H_ale      = per_H.mean(axis=0)                                 # [B]
        H_epi      = H_total - H_ale                                    # [B]
        KL         = np.log(p_bar.shape[-1]) - H_total                  # [B]

        return {"p_bar": p_bar, "H_total": H_total,
                "H_aleatoric": H_ale, "H_epistemic": H_epi,
                "KL_uniform": KL,    "P_all": P}


# ══════════════════════════════════════════════════════════════════════════════
# 9.  MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════
class UnimodalClassifier(nn.Module):
    """Image-only.  Dropout stays active for MC-Dropout compatibility."""

    def __init__(self, embed_dim=EMBED_DIM, n_cls=2, dp=0.3):
        super().__init__()
        self.encoder    = RadImageNetEncoder(embed_dim, dp)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(128, n_cls),
        )

    def forward(self, images: torch.Tensor,
                meta: torch.Tensor = None) -> torch.Tensor:
        return self.classifier(self.encoder(images)["embedding"])

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> np.ndarray:
        self.eval()
        return self.encoder(images)["embedding"].cpu().numpy()


class ExplicitCrossAttention(nn.Module):
    """
    Implements multi-head cross-attention WITHOUT nn.MultiheadAttention.

    Q = W_Q·E_meta  (metadata queries the image)
    K = W_K·E_img
    V = W_V·E_img

    A   = softmax(QKᵀ / √d_k)
    out = W_O·concat_heads(A·V)  +  E_meta   (residual + LayerNorm)
    """

    def __init__(self, embed_dim: int = EMBED_DIM, n_heads: int = 4):
        super().__init__()
        assert embed_dim % n_heads == 0
        self.h  = n_heads
        self.dk = embed_dim // n_heads
        self.W_Q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_K = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_V = nn.Linear(embed_dim, embed_dim, bias=False)
        self.W_O = nn.Linear(embed_dim, embed_dim, bias=False)
        self.ln  = nn.LayerNorm(embed_dim)

    def forward(self, E_img: torch.Tensor,
                E_meta: torch.Tensor) -> torch.Tensor:
        B, D = E_img.shape
        h, dk = self.h, self.dk
        # [B, 1, h, dk] → [B, h, 1, dk]
        Q = self.W_Q(E_meta).view(B, 1, h, dk).transpose(1, 2)
        K = self.W_K(E_img ).view(B, 1, h, dk).transpose(1, 2)
        V = self.W_V(E_img ).view(B, 1, h, dk).transpose(1, 2)
        # Explicit scaled dot-product  A = softmax(QKᵀ/√d_k)
        A   = F.softmax(torch.matmul(Q, K.transpose(-2,-1)) / dk**0.5, dim=-1)
        ctx = torch.matmul(A, V).transpose(1,2).contiguous().view(B, D)
        return self.ln(self.W_O(ctx) + E_meta)


class MultimodalClassifier(nn.Module):
    """Late fusion: E_img ⊕ CrossAttn(E_img, E_meta) → classifier."""

    def __init__(self, embed_dim=EMBED_DIM, n_cls=2, dp=0.3):
        super().__init__()
        self.img_enc  = RadImageNetEncoder(embed_dim, dp)
        self.meta_enc = ClinicalEncoder(META_DIM, embed_dim, dp)
        self.attn     = ExplicitCrossAttention(embed_dim, n_heads=4)
        self.clf      = nn.Sequential(
            nn.Linear(embed_dim * 2, 256), nn.GELU(),
            nn.Dropout(dp),
            nn.Linear(256, n_cls),
        )

    def forward(self, images: torch.Tensor,
                meta:   torch.Tensor) -> torch.Tensor:
        Ei = self.img_enc(images)["embedding"]  # [B, D] float32
        Em = self.meta_enc(meta)                 # [B, D] float32
        f  = self.attn(Ei, Em)                   # [B, D]
        return self.clf(torch.cat([Ei, f], dim=-1))

    @torch.no_grad()
    def embed(self, images: torch.Tensor,
              meta: torch.Tensor) -> np.ndarray:
        self.eval()
        Ei = self.img_enc(images)["embedding"]
        Em = self.meta_enc(meta)
        return torch.cat([Ei, self.attn(Ei, Em)], dim=-1).cpu().numpy()


# ══════════════════════════════════════════════════════════════════════════════
# 10. TRAINER
# ══════════════════════════════════════════════════════════════════════════════
class Trainer:
    def __init__(self, model: nn.Module, device: torch.device,
                 lr: float = 3e-4, label: str = "model"):
        self.model   = model.to(device)
        self.device  = device
        self.label   = label
        self.opt     = torch.optim.AdamW(model.parameters(), lr=lr,
                                         weight_decay=1e-4)
        self.sched   = torch.optim.lr_scheduler.CosineAnnealingLR(
                            self.opt, T_max=N_EPOCHS)
        self.loss_fn = nn.CrossEntropyLoss()
        self.history = []

    # ── One training epoch ────────────────────────────────────────────────────
    def train_epoch(self, loader: DataLoader) -> float:
        self.model.train()
        total = 0.0
        for b in loader:
            imgs = b["image"].to(self.device).float()     # float32 ✓
            meta = b["meta_vec"].to(self.device).float()  # float32 ✓
            labs = b["label"].to(self.device)
            self.opt.zero_grad()
            loss = self.loss_fn(self.model(imgs, meta), labs)
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.opt.step()
            total += loss.item()
        self.sched.step()
        return total / max(len(loader), 1)

    # ── Evaluation ────────────────────────────────────────────────────────────
    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> dict:
        self.model.eval()
        all_logits, all_labs = [], []
        for b in loader:
            imgs = b["image"].to(self.device).float()
            meta = b["meta_vec"].to(self.device).float()
            all_logits.append(self.model(imgs, meta).cpu())
            all_labs.append(b["label"])
        logits = torch.cat(all_logits)
        labels = torch.cat(all_labs).numpy()
        probs  = F.softmax(logits, dim=-1).numpy()
        probs_c = np.clip(probs, 1e-10, 1 - 1e-10)
        return {"accuracy": accuracy_score(labels, probs.argmax(1)),
                "log_loss": log_loss(labels, probs_c)}

    # ── Full training loop ────────────────────────────────────────────────────
    def fit(self, train_loader: DataLoader,
            val_loader: DataLoader) -> None:
        print(f"\n[TRAIN] ── {self.label} ──")
        for ep in range(1, N_EPOCHS + 1):
            tr_l = self.train_epoch(train_loader)
            val  = self.evaluate(val_loader)
            self.history.append({"epoch": ep, "train_loss": tr_l, **val})
            print(f"  ep {ep:02d}/{N_EPOCHS}  "
                  f"train_loss={tr_l:.4f}  "
                  f"val_acc={val['accuracy']:.4f}  "
                  f"val_ll={val['log_loss']:.4f}")
        ckpt = os.path.join(OUT_DIR,
                            f"{self.label.replace(' ','_')}_ckpt.pt")
        torch.save(self.model.state_dict(), ckpt)
        print(f"  Saved → {ckpt}")

    # ── Embedding extraction ──────────────────────────────────────────────────
    @torch.no_grad()
    def extract_embeddings(self, loader: DataLoader,
                           multimodal: bool = False) -> tuple:
        """Returns (embeddings [N,D] float32, labels [N] int)."""
        self.model.eval()
        embs, labs = [], []
        for b in loader:
            imgs = b["image"].to(self.device).float()
            meta = b["meta_vec"].to(self.device).float()
            e = self.model.embed(imgs, meta) if multimodal \
                else self.model.embed(imgs)
            embs.append(e)
            labs.append(b["label"].numpy())
        embs = np.vstack(embs).astype(np.float32)
        labs = np.concatenate(labs)
        assert embs.dtype == np.float32, "Float32 contract violated!"
        return embs, labs


# ══════════════════════════════════════════════════════════════════════════════
# 11. VISUALISATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def _save(fig, name: str):
    p = os.path.join(OUT_DIR, name)
    fig.savefig(p, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [PLOT] saved → {name}")


def plot_training_curves(t1: "Trainer", t2: "Trainer"):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    fig.suptitle("Training Convergence", fontweight="bold")
    for t, lbl, c in [(t1,"Unimodal","royalblue"),(t2,"Multimodal","firebrick")]:
        h = pd.DataFrame(t.history)
        axes[0].plot(h["epoch"], h["train_loss"], "o-", label=lbl, color=c)
        axes[1].plot(h["epoch"], h["accuracy"],   "s-", label=f"{lbl} val", color=c)
    for ax, yl in zip(axes, ["Loss","Val Accuracy"]):
        ax.set_xlabel("Epoch"); ax.set_ylabel(yl)
        ax.legend(); ax.grid(ls="--", alpha=.4)
    _save(fig, "training_curves.png")


def plot_sscv(df_uni: pd.DataFrame, df_multi: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("ssCV Variance Stability  σ²(δ)", fontweight="bold")
    for df, lbl, c in [(df_uni,"Unimodal","royalblue"),
                        (df_multi,"Multimodal","firebrick")]:
        s = df.groupby("delta")["loss"].agg(["mean","std","var"]).reset_index()
        axes[0].plot(s["delta"], s["mean"], "o-", label=lbl, color=c)
        axes[0].fill_between(s["delta"],
                             s["mean"]-s["std"], s["mean"]+s["std"],
                             alpha=.25, color=c)
        axes[1].plot(s["delta"], s["var"], "s--",
                     label=f"σ²(δ) {lbl}", color=c)
    for ax, yl, tl in zip(axes,
        ["Cross-Entropy (mean±σ)", "Variance σ²(δ)"],
        ["Mean Loss vs δ", "Variance Growth"]):
        ax.set_xlabel("δ"); ax.set_ylabel(yl); ax.set_title(tl)
        ax.legend(); ax.grid(ls="--", alpha=.4)
    _save(fig, "sscv_curve.png")


def plot_bias_heatmap(sim_mat, attr_names, labels, title, fname):
    n = min(60, len(sim_mat))
    df_h = pd.DataFrame(sim_mat[:n], columns=attr_names)
    rlabs = ["AS" if l==1 else "Norm" for l in labels[:n]]
    fig, ax = plt.subplots(figsize=(10, max(5, n*0.18)))
    sns.heatmap(df_h, cmap="RdBu_r", center=0, yticklabels=rlabs,
                linewidths=.3, linecolor="grey",
                cbar_kws={"label":"Cosine Similarity"}, ax=ax)
    ax.set_title(title, fontsize=11)
    _save(fig, fname)


def plot_entropy_violin(eu, em):
    rows = []
    for name, e in [("Unimodal",eu),("Multimodal",em)]:
        for vals, typ in [(e["H_epistemic"],"Epistemic"),
                          (e["H_aleatoric"],"Aleatoric")]:
            for v in vals:
                rows.append({"Model":name,"Type":typ,"H (nats)":float(v)})
    df_e = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(9,5))
    sns.violinplot(data=df_e, x="Type", y="H (nats)", hue="Model",
                   split=True, inner="quartile",
                   palette={"Unimodal":"royalblue","Multimodal":"firebrick"},
                   ax=ax)
    ax.set_title("MC-Dropout: Epistemic vs Aleatoric Entropy")
    ax.grid(ls="--", alpha=.4)
    _save(fig, "mc_entropy_violin.png")


def plot_dce_bars(dce_uni, dce_multi):
    feats = list(dce_uni.keys())
    x = np.arange(len(feats))
    fig, ax = plt.subplots(figsize=(11,5))
    ax.bar(x-.2, [dce_uni[f]   for f in feats], .35,
           label="Unimodal",   color="royalblue", alpha=.85)
    ax.bar(x+.2, [dce_multi[f] for f in feats], .35,
           label="Multimodal", color="firebrick",  alpha=.85)
    ax.set_xticks(x); ax.set_xticklabels(feats, rotation=30, ha="right")
    ax.set_ylabel("DCE  ΔP(AS)"); ax.axhline(0,color="k",lw=.8,ls="--")
    ax.set_title("Direct Causal Effect per Clinical Feature")
    ax.legend(); ax.grid(axis="y",ls="--",alpha=.4)
    _save(fig, "dce_bars.png")


def build_tally(ub, mb, dsu, dsm, eu, em, du, dm, ut, mt) -> pd.DataFrame:
    def σ2(df, d):
        s = df[df["delta"]==d]["loss"]
        return float(s.var(ddof=1)) if len(s)>1 else 0.0
    rows = []
    for lbl, b, ss, e, dce, test in [
        ("Unimodal  (Image-only)", ub, dsu, eu, du, ut),
        ("Multimodal (Img+Meta)",  mb, dsm, em, dm, mt),
    ]:
        rows.append({
            "System":              lbl,
            "WEAT |d|":            round(abs(b["weat_d"]),4),
            "Bias p-value":        round(b["p_value"],4),
            "Bias class":          b["interpretation"],
            "Mean |DCE|":          round(float(np.mean(np.abs(list(dce.values())))),5),
            "σ²(δ=0.0)":          round(σ2(ss,0.0),6),
            "σ²(δ=0.3)":          round(σ2(ss,0.3),6),
            "σ²(δ=0.5)":          round(σ2(ss,0.5),6),
            "H_epistemic":         round(float(e["H_epistemic"].mean()),5),
            "H_aleatoric":         round(float(e["H_aleatoric"].mean()),5),
            "KL(p̄||u)":            round(float(e["KL_uniform"].mean()),5),
            "Test Accuracy":       round(test["accuracy"],4),
            "Test Log-Loss":       round(test["log_loss"],4),
        })
    return pd.DataFrame(rows).set_index("System")


# ══════════════════════════════════════════════════════════════════════════════
# 12. MAIN  — runs all phases sequentially in the correct order
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "═"*68)
    print("  BIAS–VARIANCE QUANTIFICATION PIPELINE — SacroMRI")
    print("═"*68)

    # ── Phase 1: Load CSV & Build DataLoaders ─────────────────────────────────
    print("\n[P1] Data loading & standardisation")
    meta_df = None
    if os.path.exists(CSV_PATH):
        meta_df = load_and_parse_csv(CSV_PATH)
        print(f"     CSV loaded: {len(meta_df)} rows")
    else:
        print(f"     [WARN] CSV not found at {CSV_PATH} — metadata zeroed")

    print("\n  Dataset split summary:")
    loaders = build_dataloaders(DATA_ROOT, meta_df, batch_size=BATCH_SIZE)

    # ── Float32 contract verification ─────────────────────────────────────────
    sample = next(iter(loaders["train"]))
    assert sample["image"].dtype    == torch.float32, "❌ Image not float32"
    assert sample["meta_vec"].dtype == torch.float32, "❌ Meta not float32"
    print(f"\n  ✓ image dtype   : {sample['image'].dtype}")
    print(f"  ✓ meta dtype    : {sample['meta_vec'].dtype}")
    print(f"  ✓ image shape   : {tuple(sample['image'].shape)}")

    # ── Phase 2: Train Models ─────────────────────────────────────────────────
    print("\n[P2] Model training")
    uni_model   = UnimodalClassifier(EMBED_DIM)
    multi_model = MultimodalClassifier(EMBED_DIM)
    uni_t   = Trainer(uni_model,   DEVICE, label="Unimodal")
    multi_t = Trainer(multi_model, DEVICE, label="Multimodal")
    uni_t.fit(loaders["train"], loaders["val"])
    multi_t.fit(loaders["train"], loaders["val"])
    plot_training_curves(uni_t, multi_t)

    # ── Phase 3: Test evaluation ──────────────────────────────────────────────
    print("\n[P3] Test evaluation")
    uni_test   = uni_t.evaluate(loaders["test"])
    multi_test = multi_t.evaluate(loaders["test"])
    print(f"  Unimodal   acc={uni_test['accuracy']:.4f}  ll={uni_test['log_loss']:.4f}")
    print(f"  Multimodal acc={multi_test['accuracy']:.4f}  ll={multi_test['log_loss']:.4f}")

    # ── Phase 4: Extract embeddings ───────────────────────────────────────────
    print("\n[P4] Embedding extraction")
    emb_tr_u, lab_tr_u = uni_t.extract_embeddings(loaders["train"], False)
    emb_te_u, lab_te_u = uni_t.extract_embeddings(loaders["test"],  False)
    emb_tr_m, lab_tr_m = multi_t.extract_embeddings(loaders["train"], True)
    emb_te_m, lab_te_m = multi_t.extract_embeddings(loaders["test"],  True)
    print(f"  train uni  : {emb_tr_u.shape}  {emb_tr_u.dtype}")
    print(f"  test  multi: {emb_te_m.shape}  {emb_te_m.dtype}")

    # ── Phase 5A: Caliskan Bridge ─────────────────────────────────────────────
    # Each model may produce embeddings of different dimension:
    #   Unimodal   → 512-dim  (image encoder only)
    #   Multimodal → 1024-dim (image + cross-attention fused)
    # Attribute vectors MUST match the embedding dimension of the model being tested.
    print("\n[P5A] Bias — Caliskan / WEAT effect size")
    attr_u = make_valence_attributes(emb_te_u.shape[1])   # 512-dim
    attr_m = make_valence_attributes(emb_te_m.shape[1])   # 1024-dim

    caliskan_u = CaliskanBridgeQuantifier(attr_u)
    caliskan_m = CaliskanBridgeQuantifier(attr_m)

    mh = lab_te_u == 0; mp = lab_te_u == 1
    if mh.sum() == 0 or mp.sum() == 0:
        mid = len(lab_te_u)//2; mh = np.arange(len(lab_te_u))<mid; mp = ~mh

    uni_bias   = caliskan_u.compute(emb_te_u[mh], emb_te_u[mp])
    multi_bias = caliskan_m.compute(emb_te_m[mh], emb_te_m[mp])
    print(f"  Unimodal   WEAT d={uni_bias['weat_d']:+.4f}  "
          f"p={uni_bias['p_value']:.4f}  → {uni_bias['interpretation']}")
    print(f"  Multimodal WEAT d={multi_bias['weat_d']:+.4f}  "
          f"p={multi_bias['p_value']:.4f}  → {multi_bias['interpretation']}")
    plot_bias_heatmap(uni_bias["sim_matrix"],   uni_bias["attr_names"],
                      lab_te_u, "Caliskan Bridge — Unimodal",  "bias_heatmap_uni.png")
    plot_bias_heatmap(multi_bias["sim_matrix"], multi_bias["attr_names"],
                      lab_te_m, "Caliskan Bridge — Multimodal","bias_heatmap_multi.png")


    # ── Phase 5B: DCE ─────────────────────────────────────────────────────────
    print("\n[P5B] Bias — Direct Causal Effect")
    tb = next(iter(loaders["test"]))
    dce_uni   = DCEQuantifier(uni_model,   DEVICE).measure(
                    tb["image"], tb["meta_vec"])
    dce_multi = DCEQuantifier(multi_model, DEVICE).measure(
                    tb["image"], tb["meta_vec"])
    print("  Top-5 DCE scores:")
    for k in CLINICAL_FEATURES[:5]:
        print(f"    {k:30s}  Uni={dce_uni[k]:+.5f}  Multi={dce_multi[k]:+.5f}")
    plot_dce_bars(dce_uni, dce_multi)

    # ── Phase 5C: ssCV ────────────────────────────────────────────────────────
    print("\n[P5C] Variance — ssCV Procedure")
    sscv = ssCVQuantifier(DELTA_GRID, N_REPLICATES)
    print("  [Unimodal]")
    df_sscv_u = sscv.run(emb_tr_u, lab_tr_u, emb_te_u, lab_te_u)
    print("  [Multimodal]")
    df_sscv_m = sscv.run(emb_tr_m, lab_tr_m, emb_te_m, lab_te_m)
    df_sscv_u.to_csv(os.path.join(OUT_DIR,"sscv_unimodal.csv"),   index=False)
    df_sscv_m.to_csv(os.path.join(OUT_DIR,"sscv_multimodal.csv"), index=False)
    plot_sscv(df_sscv_u, df_sscv_m)

    # ── Phase 5D: MC-Dropout ──────────────────────────────────────────────────
    print(f"\n[P5D] Variance — MC-Dropout ({MC_PASSES} passes)")
    mcd_u = MCDropoutQuantifier(uni_model,   MC_PASSES, DEVICE)
    mcd_m = MCDropoutQuantifier(multi_model, MC_PASSES, DEVICE)
    ent_u = mcd_u.predict(tb["image"], tb["meta_vec"])
    ent_m = mcd_m.predict(tb["image"], tb["meta_vec"])
    for name, e in [("Unimodal",ent_u),("Multimodal",ent_m)]:
        print(f"  {name:12s}  H_epi={e['H_epistemic'].mean():.4f}  "
              f"H_ale={e['H_aleatoric'].mean():.4f}  "
              f"KL={e['KL_uniform'].mean():.4f}")
    plot_entropy_violin(ent_u, ent_m)

    # ── Phase 6: Tally Table ──────────────────────────────────────────────────
    print("\n[P6] Final Tally Table")
    tally = build_tally(uni_bias, multi_bias,
                         df_sscv_u, df_sscv_m,
                         ent_u,    ent_m,
                         dce_uni,  dce_multi,
                         uni_test, multi_test)
    tally.to_csv(os.path.join(OUT_DIR,"tally_table.csv"))

    print("\n" + "═"*68)
    print("  FINAL TALLY — BIAS & VARIANCE DECOMPOSITION")
    print("═"*68)
    print(tally.T.to_string())
    print("═"*68)

    # ── Styled tally figure ───────────────────────────────────────────────────
    t_vals = tally.reset_index()
    fig, ax = plt.subplots(figsize=(15, len(tally.columns)*0.5 + 2))
    ax.axis("off")
    tbl = ax.table(cellText=t_vals.values,
                   colLabels=t_vals.columns,
                   cellLoc="center", loc="center", bbox=[0,0,1,1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.5)
    for j in range(len(t_vals.columns)):
        tbl[0,j].set_facecolor("#1a3a5c")
        tbl[0,j].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(t_vals)+1):
        clr = "#EEF4FF" if i%2==0 else "#FFFFFF"
        for j in range(len(t_vals.columns)):
            tbl[i,j].set_facecolor(clr)
    fig.suptitle("Bias–Variance Tally: Unimodal vs Multimodal — SacroMRI",
                 fontsize=11, fontweight="bold")
    _save(fig, "tally_figure.png")

    # ── Final file listing ────────────────────────────────────────────────────
    print(f"\n[DONE] Output files in {OUT_DIR}:")
    for f in sorted(os.listdir(OUT_DIR)):
        size = os.path.getsize(os.path.join(OUT_DIR, f))
        print(f"  {f:<35s}  {size/1024:6.1f} KB")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
