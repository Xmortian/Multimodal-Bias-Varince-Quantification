 
import argparse
import os
import sys
import glob
import warnings
 
import numpy as np
import pandas as pd
 
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
 
warnings.filterwarnings("ignore")
 
# ---------------------------------------------------------------------------
# THRESHOLDS -- tune to taste. Defaults are strict on purpose.
# ---------------------------------------------------------------------------
LEAKAGE_ACC_WARN   = 0.95   # tabular-only CV accuracy above this = red flag
LEAKAGE_ACC_HARD   = 0.99   # above this = almost certainly leakage/separable
SINGLE_FEAT_WARN   = 0.90   # one feature alone separating classes this well
N_SPLITS           = 5      # CV folds
SEED               = 42
 
GREEN = "\033[92m"; YELLOW = "\033[93m"; RED = "\033[91m"; BOLD = "\033[1m"; END = "\033[0m"
def c(txt, col):  # color helper (degrades gracefully if piped to a file)
    return f"{col}{txt}{END}" if sys.stdout.isatty() else txt
 
 
# ===========================================================================
# Encoding helpers
# ===========================================================================
def encode_label(series):
    """Map a label column to integers. Handles yes/no, pos/neg, AS/Normal..."""
    s = series.astype(str).str.strip().str.lower()
    uniq = sorted(s.unique())
    mapping = {v: i for i, v in enumerate(uniq)}
    y = s.map(mapping).to_numpy()
    return y, mapping
 
 
def encode_features(df, cols):
    """
    Turn the chosen feature columns into a clean float matrix.
    - yes/no/positive/negative/true/false -> 1/0
    - numeric strings -> float
    - other categoricals -> one-hot
    Returns (X, feature_names).
    """
    _BOOL = {"yes": 1.0, "no": 0.0, "positive": 1.0, "negative": 0.0,
             "pos": 1.0, "neg": 0.0, "true": 1.0, "false": 0.0,
             "1": 1.0, "0": 0.0, "y": 1.0, "n": 0.0}
    mats, names = [], []
    for col in cols:
        raw = df[col]
        s = raw.astype(str).str.strip().str.lower()
        if s.isin(_BOOL.keys()).mean() > 0.95:
            mats.append(s.map(_BOOL).fillna(0.0).to_numpy().reshape(-1, 1))
            names.append(col)
            continue
        num = pd.to_numeric(raw, errors="coerce")
        if num.notna().mean() > 0.80:
            v = num.fillna(num.median())
            mats.append(v.to_numpy().reshape(-1, 1))
            names.append(col)
            continue
        # low-cardinality categorical -> one-hot; else skip
        if raw.nunique() <= 15:
            dummies = pd.get_dummies(raw.astype(str), prefix=col)
            mats.append(dummies.to_numpy().astype(float))
            names.extend(list(dummies.columns))
        else:
            print(c(f"    [skip] '{col}' is high-cardinality non-numeric "
                    f"({raw.nunique()} values) -- excluded.", YELLOW))
    if not mats:
        return None, []
    return np.hstack(mats).astype(np.float32), names
 
 
# ===========================================================================
# TEST 1 -- tabular-only separability (the leakage test)
# ===========================================================================
def test_tabular_separability(X, y):
    print(c("\n[TEST 1] Tabular-only separability  (leakage check)", BOLD))
    print("-" * 70)
    if X is None or X.shape[1] == 0:
        print(c("  No usable tabular features -- skipping.", YELLOW))
        return None
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    acc = scores.mean()
    print(f"  Linear model, tabular features ONLY (no images):")
    print(f"  {N_SPLITS}-fold CV accuracy = {acc:.4f}  (+/- {scores.std():.4f})")
    print(f"  Class balance: {np.bincount(y)}  "
          f"(majority baseline = {np.bincount(y).max()/len(y):.4f})")
    if acc >= LEAKAGE_ACC_HARD:
        print(c(f"  >>> FAIL: tabular features alone are essentially perfect "
                f"(>= {LEAKAGE_ACC_HARD}).", RED))
        print(c("      The tabular branch can solve the task with no images. "
                "Multimodal fusion is illusory here, and every bias-variance "
                "metric will be read off a saturated (loss~0) point.", RED))
    elif acc >= LEAKAGE_ACC_WARN:
        print(c(f"  >>> WARN: tabular-only accuracy is very high "
                f"(>= {LEAKAGE_ACC_WARN}). Likely strong leakage. Inspect "
                f"Test 2 to see which feature(s) are responsible.", YELLOW))
    else:
        print(c("  >>> OK: tabular features alone do NOT trivially solve the "
                "task. Room for images to contribute -> genuine multimodality "
                "is possible.", GREEN))
    return acc
 
 
# ===========================================================================
# TEST 2 -- per-feature label-proxy check
# ===========================================================================
def test_single_features(df, feature_cols, y):
    print(c("\n[TEST 2] Per-feature label-proxy check", BOLD))
    print("-" * 70)
    print("  How well does EACH single feature separate the classes?")
    print("  (A feature near 1.0 is a label proxy -- a prime leakage suspect.)\n")
    flagged = []
    for col in feature_cols:
        Xi, names = encode_features(df, [col])
        if Xi is None:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, random_state=SEED))
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
        try:
            acc = cross_val_score(clf, Xi, y, cv=cv, scoring="accuracy").mean()
        except Exception:
            continue
        tag = ""
        if acc >= SINGLE_FEAT_WARN:
            tag = c("  <-- LABEL PROXY (leakage suspect)", RED)
            flagged.append((col, acc))
        print(f"    {col:30s}  acc = {acc:.4f}{tag}")
    if flagged:
        print(c(f"\n  >>> {len(flagged)} feature(s) individually separate the "
                f"classes >= {SINGLE_FEAT_WARN}:", RED))
        for col, acc in flagged:
            print(c(f"        - {col}  ({acc:.4f})", RED))
        print(c("      Drop these (they encode the label), then re-run Test 1 "
                "on what remains. If nothing usable survives, the dataset is "
                "unsuitable for multimodal fusion research.", RED))
    else:
        print(c("\n  >>> OK: no single feature is a near-perfect label proxy.",
                GREEN))
    return flagged
 
 
# ===========================================================================
# TEST 3 -- image-only low-level-statistics baseline (optional)
# ===========================================================================
def test_image_shortcut(image_root, max_per_class=200):
    print(c("\n[TEST 3] Image-only low-level-statistics baseline", BOLD))
    print("-" * 70)
    try:
        from PIL import Image
    except ImportError:
        print(c("  Pillow not installed -- skipping image test. "
                "(pip install pillow)", YELLOW))
        return None
 
    # expect folder-per-class layout: image_root/<class>/*.png|jpg
    class_dirs = [d for d in sorted(glob.glob(os.path.join(image_root, "*")))
                  if os.path.isdir(d)]
    if len(class_dirs) < 2:
        print(c(f"  Expected >=2 class subfolders in {image_root}; "
                f"found {len(class_dirs)}. Skipping.", YELLOW))
        return None
 
    feats, labs = [], []
    for lab, d in enumerate(class_dirs):
        paths = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"):
            paths += glob.glob(os.path.join(d, ext))
        paths = sorted(paths)[:max_per_class]
        print(f"    class '{os.path.basename(d)}': sampling {len(paths)} images")
        for p in paths:
            try:
                arr = np.asarray(Image.open(p).convert("L").resize((64, 64)),
                                 dtype=np.float32) / 255.0
            except Exception:
                continue
            # cheap, content-agnostic descriptors: intensity histogram + moments
            hist, _ = np.histogram(arr, bins=16, range=(0, 1), density=True)
            stats = [arr.mean(), arr.std(),
                     np.percentile(arr, 25), np.percentile(arr, 75)]
            feats.append(np.concatenate([hist, stats]))
            labs.append(lab)
 
    if len(set(labs)) < 2 or len(labs) < 2 * N_SPLITS:
        print(c("  Not enough images loaded for a CV test. Skipping.", YELLOW))
        return None
 
    X = np.vstack(feats).astype(np.float32)
    y = np.array(labs)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=2000, random_state=SEED))
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    acc = cross_val_score(clf, X, y, cv=cv, scoring="accuracy").mean()
    print(f"\n  Linear model on LOW-LEVEL image stats only "
          f"(histogram + moments):")
    print(f"  {N_SPLITS}-fold CV accuracy = {acc:.4f}  "
          f"(majority baseline = {np.bincount(y).max()/len(y):.4f})")
    if acc >= LEAKAGE_ACC_WARN:
        print(c("  >>> WARN: classes are separable from crude global image "
                "statistics alone. The two classes probably come from "
                "different acquisition/source distributions, so a CNN may be "
                "learning a scanner-detector, not pathology.", YELLOW))
    else:
        print(c("  >>> OK: low-level stats do not separate the classes. The "
                "visual task is plausibly about content, not acquisition "
                "artifacts. (A real CNN may still find shortcuts -- this is a "
                "cheap screen, not a guarantee.)", GREEN))
    return acc
 
 
# ===========================================================================
# MAIN
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Sanity-check a dataset before committing to training.")
    ap.add_argument("--csv", required=True, help="Path to the clinical/tabular CSV.")
    ap.add_argument("--label-col", default=None,
                    help="Name of the label/diagnosis column.")
    ap.add_argument("--feature-cols", nargs="*", default=None,
                    help="Tabular feature columns to test. If omitted, the "
                         "script guesses (and warns).")
    ap.add_argument("--image-root", default=None,
                    help="Optional: folder with one subfolder per class.")
    ap.add_argument("--max-per-class", type=int, default=200,
                    help="Cap images sampled per class in Test 3.")
    args = ap.parse_args()
 
    print(c("=" * 70, BOLD))
    print(c("  DATASET SANITY CHECK", BOLD))
    print(c("=" * 70, BOLD))
    print(f"  CSV: {args.csv}")
 
    if not os.path.exists(args.csv):
        print(c(f"  CSV not found: {args.csv}", RED)); sys.exit(1)
    df = pd.read_csv(args.csv)
    df.columns = [str(col).strip() for col in df.columns]
    print(f"  Rows: {len(df)}   Columns: {list(df.columns)}")
 
    # --- resolve label column ---
    label_col = args.label_col
    if label_col is None:
        cands = [col for col in df.columns
                 if any(k in col.lower()
                        for k in ("diagnos", "label", "class", "target", "outcome"))]
        if not cands:
            print(c("  Could not auto-detect a label column. Pass --label-col.",
                    RED)); sys.exit(1)
        label_col = cands[0]
        print(c(f"  [GUESS] Using label column '{label_col}'. "
                f"Confirm with --label-col if wrong.", YELLOW))
    if label_col not in df.columns:
        print(c(f"  Label column '{label_col}' not in CSV.", RED)); sys.exit(1)
 
    y, mapping = encode_label(df[label_col])
    print(f"  Label '{label_col}' -> {mapping}")
    if len(set(y)) < 2:
        print(c("  Label has <2 classes -- nothing to test.", RED)); sys.exit(1)
 
    # --- resolve feature columns ---
    feature_cols = args.feature_cols
    if not feature_cols:
        feature_cols = [col for col in df.columns if col != label_col
                        and not any(k in col.lower()
                                    for k in ("id", "image", "path", "file"))]
        print(c(f"  [GUESS] No --feature-cols given. Using {len(feature_cols)} "
                f"columns and GUESSING -- auto-detection is exactly how leakage "
                f"slips through. Prefer naming them explicitly.", YELLOW))
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(c(f"  These --feature-cols are not in the CSV: {missing}", RED))
        sys.exit(1)
    print(f"  Testing features: {feature_cols}")
 
    X, names = encode_features(df, feature_cols)
    if X is not None:
        print(f"  Encoded feature matrix: {X.shape}")
 
    # --- run tests ---
    tab_acc = test_tabular_separability(X, y)
    flagged = test_single_features(df, feature_cols, y)
    img_acc = (test_image_shortcut(args.image_root, args.max_per_class)
               if args.image_root else None)
 
    # --- verdict ---
    print(c("\n" + "=" * 70, BOLD))
    print(c("  VERDICT", BOLD))
    print(c("=" * 70, BOLD))
    fail = (tab_acc is not None and tab_acc >= LEAKAGE_ACC_HARD)
    warn = (
        (tab_acc is not None and LEAKAGE_ACC_WARN <= tab_acc < LEAKAGE_ACC_HARD)
        or bool(flagged)
        or (img_acc is not None and img_acc >= LEAKAGE_ACC_WARN)
    )
    if fail:
        print(c("  REJECT: tabular features alone solve the task. Unsuitable "
                "for a multimodal bias-variance study without major surgery "
                "(drop proxy features and re-test, or pick another dataset).",
                RED))
    elif warn:
        print(c("  CAUTION: leakage or shortcut signals present. Address the "
                "flagged items and re-run before training. Do NOT start 17 "
                "backbones yet.", YELLOW))
    else:
        print(c("  PROCEED (with eyes open): no obvious leakage or shortcut. "
                "This is necessary, not sufficient -- confirm models actually "
                "land below the ceiling (loss>0, acc<1) once you train.",
                GREEN))
    print(c("=" * 70 + "\n", BOLD))
 
 
if __name__ == "__main__":
    main()
