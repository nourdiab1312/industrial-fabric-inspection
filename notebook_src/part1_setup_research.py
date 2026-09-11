"""Sections 00-03: configuration, research question, literature, dataset investigation."""

CELLS = []
md = lambda s: CELLS.append(("markdown", s))
code = lambda s: CELLS.append(("code", s))

md(r"""# Industrial Fabric Inspection Research

**Automated visual quality inspection for continuous textile production — research + software prototype**

> Everything in this notebook runs on Google Colab with public datasets and **software-based production simulation**.
> Nothing here is factory validation. Claims are labelled as *Demonstrated / Suggested / Hypothesized / Future work*.

Working loop used throughout: **Explain → Implement → Run → Inspect → Interpret → Decide → Next step**

Each section follows the same template: Objective · Research Question · Method · Implementation · Results · Interpretation · Limitations · Decision.

**How to use this notebook.** Section 00 defines `RUN` flags. Sections are self-contained and gated by those flags so the notebook can be executed phase by phase without running every experiment. Results are persisted to `RESULTS_DIR` so later sections read saved artefacts instead of relying on hidden kernel state.
""")

md(r"""## 00. Project Configuration

### Objective
Make every experiment reproducible: seeds, paths, hardware detection, package versions, and a single experiment tracker.

### Why
Research results that cannot be reproduced cannot support a paper or a product decision. Colab sessions are ephemeral, so all artefacts are written to a persistent root (Google Drive when mounted, `/content` otherwise).
""")

code(r'''# @title 00.1 Install dependencies (idempotent)
import importlib, subprocess, sys

def pip_install(pkgs):
    missing = []
    for spec in pkgs:
        mod = spec.split("==")[0].split(">=")[0].replace("-", "_")
        mod = {"scikit_learn": "sklearn", "opencv_python_headless": "cv2",
               "segmentation_models_pytorch": "segmentation_models_pytorch",
               "imagehash": "imagehash", "ultralytics": "ultralytics", "timm": "timm",
               "faiss_cpu": "faiss", "pycocotools": "pycocotools"}.get(mod, mod)
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(spec)
    if missing:
        print("Installing:", missing)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", *missing], check=False)
    else:
        print("All packages already available.")

pip_install([
    "timm", "ultralytics", "segmentation-models-pytorch", "imagehash",
    "pycocotools", "faiss-cpu", "opencv-python-headless", "scikit-learn", "pandas", "matplotlib", "seaborn"
])
''')

code(r'''# @title 00.2 Global configuration
import os, json, random, time, platform, datetime
from pathlib import Path
import numpy as np
import torch

SEED = 42
IMG_SIZE = 256          # default classification / anomaly input size (ablated in Section 19)
BATCH_SIZE = 32
NUM_WORKERS = 2
LEARNING_RATE = 3e-4
NUM_EPOCHS = 10
USE_AMP = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Phase gates: nothing expensive runs unless its flag is True ----
RUN = dict(
    ACQUIRE_DATA      = False,   # 04
    AUDIT             = False,   # 05
    EDA               = False,   # 06
    LEAKAGE           = False,   # 07
    PREPARE           = False,   # 08
    CLASSIFICATION    = False,   # 09
    DETECTION         = False,   # 10
    SEGMENTATION      = False,   # 11
    ANOMALY           = False,   # 12
    TEMPORAL          = False,   # 13
    CROSS_DATASET     = False,   # 14
    SEVERITY          = False,   # 15
    DECISION_ENGINE   = False,   # 16
    SIMULATION        = False,   # 17
    ERROR_ANALYSIS    = False,   # 18
    ABLATION          = False,   # 19
    EFFICIENCY        = False,   # 20
)

# ---- Persistent storage: Google Drive if available, else /content ----
def resolve_project_root():
    try:
        from google.colab import drive  # noqa
        drive.mount("/content/drive", force_remount=False)
        return Path("/content/drive/MyDrive/Industrial_Fabric_Inspection")
    except Exception:
        return Path("/content/Industrial_Fabric_Inspection")

PROJECT_ROOT = resolve_project_root()
RAW_DIR       = PROJECT_ROOT / "data" / "RAW"          # never modified
PROCESSED_DIR = PROJECT_ROOT / "data" / "PROCESSED"    # derived, reproducible
SPLITS_DIR    = PROJECT_ROOT / "data" / "SPLITS"       # csv split manifests
RESULTS_DIR   = PROJECT_ROOT / "results"
CKPT_DIR      = PROJECT_ROOT / "checkpoints"
FIG_DIR       = RESULTS_DIR / "figures"
for d in [RAW_DIR, PROCESSED_DIR, SPLITS_DIR, RESULTS_DIR, CKPT_DIR, FIG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()
print("PROJECT_ROOT:", PROJECT_ROOT)
print("DEVICE:", DEVICE)
''')

code(r'''# @title 00.3 Hardware + environment record
import importlib.metadata as md_

def env_report():
    rep = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        rep.update(gpu_name=p.name, gpu_mem_gb=round(p.total_memory / 1e9, 2))
    for pkg in ["torchvision", "timm", "ultralytics", "segmentation_models_pytorch",
                "opencv-python-headless", "scikit-learn", "numpy", "pandas", "imagehash"]:
        try:
            rep[pkg] = md_.version(pkg)
        except Exception:
            rep[pkg] = "not installed"
    return rep

ENV = env_report()
print(json.dumps(ENV, indent=2))
(RESULTS_DIR / "environment.json").write_text(json.dumps(ENV, indent=2))
''')

code(r'''# @title 00.4 Experiment tracker (single source of truth for all results)
import pandas as pd

EXPERIMENT_LOG = RESULTS_DIR / "experiments.csv"
EXPERIMENT_COLUMNS = [
    "experiment_id", "date", "dataset", "split", "task", "model", "input_size", "augmentation",
    "hyperparameters", "training_time_s", "inference_time_ms", "accuracy", "precision", "recall",
    "f1", "map50", "map50_95", "iou", "dice", "auroc", "auprc", "fnr", "fpr", "fps", "params_m",
    "model_size_mb", "notes", "decision",
]

def load_experiments():
    if EXPERIMENT_LOG.exists():
        return pd.read_csv(EXPERIMENT_LOG)
    return pd.DataFrame(columns=EXPERIMENT_COLUMNS)

def log_experiment(**kwargs):
    """Append one experiment row. Only fill metrics relevant to the task; leave others NaN."""
    df = load_experiments()
    row = {c: kwargs.get(c, np.nan) for c in EXPERIMENT_COLUMNS}
    row["date"] = row["date"] if isinstance(row["date"], str) else datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    if not isinstance(row["experiment_id"], str):
        row["experiment_id"] = f"EXP-{len(df) + 1:03d}"
    if isinstance(row["hyperparameters"], dict):
        row["hyperparameters"] = json.dumps(row["hyperparameters"], sort_keys=True)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(EXPERIMENT_LOG, index=False)
    print(f"Logged {row['experiment_id']} — {row['task']} / {row['model']} on {row['dataset']}")
    return row["experiment_id"]

def save_json(obj, name):
    p = RESULTS_DIR / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    return p

def load_json(name, default=None):
    p = RESULTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else default

print(load_experiments().tail())
''')

code(r'''# @title 00.5 Shared utilities (imaging, timing, plotting)
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import contextmanager

sns.set_theme(style="whitegrid", context="notebook")
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def list_images(root):
    root = Path(root)
    return sorted(p for p in root.rglob("*") if p.suffix.lower() in IMG_EXTS)

def read_image(path, gray=False):
    flag = cv2.IMREAD_GRAYSCALE if gray else cv2.IMREAD_COLOR
    img = cv2.imread(str(path), flag)
    if img is None:
        raise IOError(f"Unreadable image: {path}")
    return img if gray else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

@contextmanager
def timer(label="block"):
    t0 = time.perf_counter()
    yield
    print(f"[{label}] {time.perf_counter() - t0:.2f}s")

def savefig(name, dpi=120):
    p = FIG_DIR / name
    plt.savefig(p, dpi=dpi, bbox_inches="tight")
    print("saved", p)

def show_grid(paths, titles=None, ncols=4, size=3, gray=False, masks=None):
    n = len(paths); nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * size, nrows * size))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        img = read_image(paths[i], gray=gray)
        ax.imshow(img, cmap="gray" if gray else None)
        if masks is not None and masks[i] is not None:
            m = read_image(masks[i], gray=True)
            ax.imshow(np.ma.masked_where(m == 0, m), cmap="autumn", alpha=0.5)
        if titles:
            ax.set_title(str(titles[i]), fontsize=8)
    plt.tight_layout()
    return fig

def count_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6

def model_size_mb(model):
    return sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6

@torch.no_grad()
def benchmark_latency(model, input_shape, n_warmup=10, n_iters=50, device=DEVICE):
    model.eval().to(device)
    x = torch.randn(1, *input_shape, device=device)
    for _ in range(n_warmup):
        model(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        model(x)
        if device.type == "cuda":
            torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_iters * 1000
    return {"latency_ms": ms, "fps": 1000 / ms}
''')

md(r"""## 01. Research Question

### Problem definition
Textile mills inspect fabric for defects (holes, broken yarns, stains, oil marks, misweaves, etc.) largely by human operators watching moving fabric. Human inspection reaches roughly 60–75 % detection rates in the literature and degrades with fatigue *(this range is repeatedly cited in fabric-inspection surveys; exact numbers vary by study — requires verification per source)*. The industrial goal is **continuous, in-line, automated inspection** that produces a quality decision before packaging.

### What the system is
```
Continuous production → camera → frame sequence → AI inspection → localisation → type/anomaly → severity → PASS / REVIEW / REJECT → inspection record
```

### What the system is NOT
A single uploaded photo classified as defective / not defective.

### Primary research question (current formulation)
> **How robust is automated fabric-defect inspection when moving from a static, single-dataset benchmark setting to a continuous-production setting with sequential frames and acquisition-domain shift, and does exploiting temporal information and anomaly detection improve robustness compared with independent-frame supervised inspection?**

### Sub-questions
| ID | Sub-question | Sections |
|----|--------------|----------|
| RQ1 | What is a defensible supervised baseline (classification/detection/segmentation) on production-derived data with leakage-safe splits? | 07–11 |
| RQ2 | When defects are rare and diverse, is one-class anomaly detection more robust than supervised inspection on the same test set? | 12, 21 |
| RQ3 | Does temporal aggregation over consecutive frames reduce false alarms / missed defects versus frame-independent inspection? | 13, 17 |
| RQ4 | How much does performance degrade under cross-dataset (camera / fabric / illumination) shift, and why? | 14 |
| RQ5 | How does defect size affect detection performance, and does resolution or tiling help? | 10, 18, 19 |
| RQ6 | Can model outputs be converted into an explainable PASS/REVIEW/REJECT decision with experimentally derived thresholds? | 15–16 |
| RQ7 | What accuracy/latency trade-off is realistic for near-line inference? | 20–21 |

*These are hypotheses to be tested, not conclusions.*
""")

md(r"""## 02. Literature Investigation

### Objective
Build a literature matrix of ~20–30 papers so that the research gap **emerges from evidence** rather than being assumed.

### Method
The matrix below is stored as a DataFrame and saved to `results/literature_matrix.csv`. Entries are seeded with the works that are directly relevant (dataset papers + key method/survey papers). Each row has a `verified` column:

* `verified = True` — title/venue/main claim checked against the original source during this project.
* `verified = False` — entry added from memory or secondary citation; **requires verification before it is cited in any write-up.**

**Rule:** do not fabricate author lists, exact metric values, or venues. Where unknown, write `Unknown — requires verification`. Extend the table as papers are read (`add_paper(...)`).
""")

code(r'''# @title 02.1 Literature matrix
UNK = "Unknown — requires verification"

LIT_COLUMNS = ["title", "year", "authors", "venue", "research_problem", "industrial_scenario", "dataset",
               "dataset_size", "acquisition", "camera", "static_or_temporal", "task", "model", "training_strategy",
               "metrics", "realtime_eval", "hardware", "external_validation", "main_result", "main_limitation",
               "research_gap", "relevance", "verified"]

LIT_PATH = RESULTS_DIR / "literature_matrix.csv"

def load_lit():
    return pd.read_csv(LIT_PATH) if LIT_PATH.exists() else pd.DataFrame(columns=LIT_COLUMNS)

def add_paper(**kw):
    df = load_lit()
    row = {c: kw.get(c, UNK) for c in LIT_COLUMNS}
    if row["title"] in set(df["title"]):
        df = df[df["title"] != row["title"]]
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(LIT_PATH, index=False)
    return df

seed_papers = [
    dict(title="TSFabrics: a time-series dataset for real-time fabric defect detection on circular knitting machines",
         year=2026, venue=UNK, research_problem="Static image datasets ignore the temporal continuity of production",
         industrial_scenario="Circular knitting machines, continuous production", dataset="TSFabrics",
         dataset_size="93,196 grayscale images, 800x640, 22 production scenarios (per dataset description)",
         acquisition="In-line camera on knitting machine; varied lighting/speed/fabric", camera=UNK,
         static_or_temporal="Temporal (frame sequences)", task="Detection / segmentation, pixel-level labels incl. non-defect cut lines",
         realtime_eval="Claimed focus", external_validation=UNK,
         main_result="Provides a temporal benchmark; baselines reported in paper (values require verification)",
         main_limitation="Single machine type (circular knitting); grayscale only",
         research_gap="Temporal consistency vs frame-independent detection", relevance="Primary candidate for RQ3", verified=False),
    dict(title="RAW-FABRID: annotated dataset of raw industrial fabric images for anomaly detection and segmentation",
         year=UNK, venue="Mendeley Data (dataset) — paper venue requires verification",
         research_problem="Lack of realistic raw-fabric imagery with line-scan acquisition", industrial_scenario="Raw (greige) fabric inspection",
         dataset="RAW-FABRID", dataset_size="709 images, 1792x1024, 8-bit grayscale + MVTec-style patch subset",
         acquisition="Basler line-scan camera, controlled LED illumination", camera="Basler line-scan",
         static_or_temporal="Static (line-scan strips)", task="Anomaly detection, semantic segmentation (COCO JSON + CSV metadata)",
         main_limitation="Small image count; single acquisition setup", research_gap="Small data, high resolution, small defects",
         relevance="Segmentation + small-defect + cross-domain target", verified=False),
    dict(title="Lusitano: an industrial fabric anomaly-detection dataset",
         year=UNK, venue=UNK, research_problem="Industrial one-class anomaly detection on real production fabric",
         industrial_scenario="Portuguese textile company production", dataset="Lusitano",
         dataset_size="~32,000 defect-free train images, 4096x1024; test = normal + defective", acquisition="Teledyne Dalsa Linea line-scan",
         camera="Teledyne Dalsa Linea", static_or_temporal="Static strips (sequential origin possible — verify)",
         task="One-class anomaly detection (MVTec AD folder structure)", main_limitation="Normal-heavy; defect diversity per paper requires verification",
         research_gap="Scale of normal data vs rarity of defects", relevance="Primary anomaly-detection dataset candidate", verified=False),
    dict(title="ZJU-Leaper: A Benchmark Dataset for Fabric Defect Detection and a Comparative Study",
         year=2020, venue="IEEE Transactions on Artificial Intelligence (requires verification)", research_problem="Lack of large, realistic fabric benchmark",
         industrial_scenario="Industrial assembly line", dataset="ZJU-Leaper", dataset_size="~98k images across 19 fabric groups (verify)",
         static_or_temporal="Static", task="Classification / detection / segmentation, annotation-free to supervised",
         main_limitation="Static; annotation quality varies by group", relevance="Large-scale benchmark + generalization", verified=False),
    dict(title="AITEX fabric image database (public textile defect dataset)", year=2019, venue="Autex Research Journal (verify)",
         research_problem="Public benchmark for fabric defect segmentation", dataset="AITEX", dataset_size="245 images, 4096x256, 7 fabric types, 140 normal / 105 defective, 12 defect types",
         static_or_temporal="Static", task="Segmentation (pixel masks)", main_limitation="Very small; risk of overfitting and leakage",
         relevance="Historical benchmark / external validation only", verified=False),
    dict(title="TILDA Textile Texture Database", year=1996, venue="DFG Textile Texture project (verify)", dataset="TILDA",
         dataset_size="3,200 grayscale images, 8 texture classes, 7 defect classes (plus derivative subsets)", static_or_temporal="Static",
         task="Classification (derivatives add boxes)", main_limitation="Old, low resolution, lab acquisition", relevance="Literature comparison only", verified=False),
    dict(title="Chenab Textile fabric defect dataset (production-line imagery, Pakistan)", year=UNK, venue=UNK, dataset="Chenab Textile",
         dataset_size="Unknown — requires verification; 7 defect classes reported", acquisition="Captured on production lines, plain + printed fabric",
         static_or_temporal="Static", task="Detection (YOLO-style studies reported)", main_limitation="Provenance/licence via shortened link — verify",
         relevance="External validation / domain shift", verified=False),
    dict(title="PatchCore: Towards Total Recall in Industrial Anomaly Detection", year=2022, venue="CVPR", authors="Roth et al.",
         research_problem="Cold-start industrial anomaly detection", dataset="MVTec AD", task="Anomaly detection", model="PatchCore (memory bank of patch features)",
         metrics="Image/pixel AUROC", main_result="State-of-the-art image AUROC on MVTec AD (exact value verify)", main_limitation="Memory grows with train set; textures vs objects",
         relevance="Anomaly baseline (Section 12)", verified=False),
    dict(title="PaDiM: a Patch Distribution Modeling Framework for Anomaly Detection and Localization", year=2021, venue="ICPR workshops", authors="Defard et al.",
         dataset="MVTec AD", task="Anomaly detection", model="PaDiM (per-position Gaussian on CNN features)", relevance="Lightweight anomaly baseline", verified=False),
    dict(title="MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection", year=2019, venue="CVPR", authors="Bergmann et al.",
         dataset="MVTec AD", task="Anomaly detection benchmark protocol", relevance="Protocol template for one-class evaluation", verified=False),
    dict(title="Fabric defect detection survey (deep-learning era)", year=UNK, venue=UNK,
         research_problem="Taxonomy of fabric-inspection methods", main_limitation="Surveys rarely evaluate cross-dataset or temporal settings",
         research_gap="External validation + temporal inspection under-studied", relevance="Gap evidence — must be replaced by specific verified surveys", verified=False),
]
for p in seed_papers:
    LIT = add_paper(**p)

pd.set_option("display.max_colwidth", 60)
print(f"{len(LIT)} papers in matrix; {LIT['verified'].sum()} verified.")
LIT[["title", "year", "dataset", "static_or_temporal", "task", "verified"]]
''')

md(r"""### 02.2 Emerging themes (hypotheses, not conclusions)

Reading across the seeded matrix, the following gaps appear repeatedly and become **candidate** research directions to be confirmed as the matrix grows:

1. **Static evaluation dominates.** Almost all benchmark work treats frames as independent images; only TSFabrics explicitly provides temporal sequences.
2. **In-domain reporting.** Papers report train/test on the same dataset; cross-dataset degradation is rarely quantified.
3. **No decision layer.** Papers stop at mAP/AUROC; there is little work turning outputs into PASS/REVIEW/REJECT decisions with justified thresholds.
4. **Small defects and high resolution.** Line-scan data (RAW-FABRID, Lusitano) has very small defect-to-image ratios; tiling/resolution studies are sparse.
5. **Supervised vs one-class comparisons under identical protocols are uncommon.**

> Action: keep adding papers with `add_paper(...)`; flip `verified=True` only after reading the primary source. Themes are re-assessed in Section 23.
""")

md(r"""## 03. Dataset Investigation & Selection

### Objective
Select datasets by fit to the research questions, **not popularity**. AITEX is a candidate benchmark only.

### Method
1. Build the dataset comparison matrix (all 7 candidates, `Unknown — requires verification` where unsure).
2. Score each dataset on 12 criteria (0–3) with a written justification per score.
3. Recommend primary / secondary / external-validation / benchmark roles.
""")

code(r'''# @title 03.1 Dataset comparison matrix
DATASETS = {
 "TSFabrics": dict(year=2026, official_source="Paper/dataset page — record exact URL at acquisition", license=UNK,
    image_count="93,196", normal_count=UNK, defective_count=UNK, defect_categories="Defects + non-defect cut lines (pixel-level)",
    resolution="800x640", color="grayscale", static_or_sequential="sequential", continuous_production="yes",
    production_line_acquisition="yes (circular knitting machines)", camera=UNK, line_scan=UNK,
    lighting="varied across 22 scenarios", manufacturer_count=UNK, fabric_types="multiple (22 scenarios)",
    bboxes="derivable from masks", masks="yes", pixel_labels="yes", metadata="scenario/time (verify)",
    annotation_quality=UNK, known_limitations="single machine class; grayscale", class_imbalance="likely normal-heavy (verify)",
    potential_leakage="HIGH: consecutive frames nearly identical -> must split by scenario/sequence",
    train_test_protocol=UNK, public="yes (verify access)", download_size=UNK),
 "RAW-FABRID": dict(year=UNK, official_source="Mendeley Data", license="CC BY 4.0 (typical for Mendeley — verify)",
    image_count="709", normal_count=UNK, defective_count=UNK, defect_categories=UNK, resolution="1792x1024",
    color="8-bit grayscale", static_or_sequential="static strips", continuous_production="acquisition from line-scan; sequence order unknown",
    production_line_acquisition="industrial acquisition rig", camera="Basler line-scan", line_scan="yes",
    lighting="controlled LED", manufacturer_count=UNK, fabric_types="raw/greige textiles", bboxes="COCO JSON",
    masks="yes (segmentation)", pixel_labels="yes", metadata="CSV", annotation_quality=UNK,
    known_limitations="small; single rig", class_imbalance=UNK, potential_leakage="patch subset derived from full images -> patch/parent leakage",
    train_test_protocol="MVTec-style patch split provided", public="yes", download_size=UNK),
 "Lusitano": dict(year=UNK, official_source="https://kailashhambarde.github.io/Lusitano/", license=UNK,
    image_count="~32,000 train (normal) + test set", normal_count="~32,000 train", defective_count=UNK,
    defect_categories=UNK, resolution="4096x1024", color=UNK, static_or_sequential="strips (order unknown)",
    continuous_production="derived from production", production_line_acquisition="yes", camera="Teledyne Dalsa Linea",
    line_scan="yes", lighting=UNK, manufacturer_count="1 (Paulo de Oliveira S.A.)", fabric_types=UNK,
    bboxes=UNK, masks="ground-truth masks for test (MVTec layout — verify)", pixel_labels="verify", metadata=UNK,
    annotation_quality=UNK, known_limitations="single company; normal-heavy", class_imbalance="extreme (one-class)",
    potential_leakage="same roll across train/test unknown", train_test_protocol="MVTec AD layout", public="yes (research)", download_size="large — verify"),
 "Chenab Textile": dict(year=UNK, official_source="shortened link in papers — verify original host", license=UNK,
    image_count=UNK, normal_count=UNK, defective_count=UNK, defect_categories="7 (baekra, colour, contamination, cut, gray stitching, selvet, stain)",
    resolution=UNK, color="colour (plain + printed)", static_or_sequential="static", continuous_production="captured on production lines",
    production_line_acquisition="yes", camera=UNK, line_scan=UNK, lighting="varied industrial", manufacturer_count="1",
    fabric_types="plain + printed", bboxes="yes (YOLO studies)", masks=UNK, pixel_labels=UNK, metadata=UNK,
    annotation_quality=UNK, known_limitations="provenance/licence unclear", class_imbalance=UNK,
    potential_leakage=UNK, train_test_protocol=UNK, public="reported yes — verify", download_size=UNK),
 "ZJU-Leaper": dict(year=2020, official_source="Project page (verify URL)", license=UNK, image_count="~98k (verify)",
    normal_count=UNK, defective_count=UNK, defect_categories="multiple; 19 fabric groups (verify)", resolution=UNK,
    color="colour", static_or_sequential="static", continuous_production="no", production_line_acquisition="assembly-line setting (verify)",
    camera=UNK, line_scan=UNK, lighting=UNK, manufacturer_count=UNK, fabric_types="19 groups (verify)",
    bboxes="yes", masks="yes", pixel_labels="yes", metadata="fabric group", annotation_quality=UNK,
    known_limitations="static; access requires request (verify)", class_imbalance=UNK, potential_leakage="same fabric across splits",
    train_test_protocol="official splits provided", public="on request (verify)", download_size=UNK),
 "TILDA": dict(year=1996, official_source="Univ. Freiburg / DFG (verify)", license=UNK, image_count="3,200",
    normal_count=UNK, defective_count=UNK, defect_categories="7 + normal", resolution="768x512 (verify)", color="grayscale",
    static_or_sequential="static", continuous_production="no", production_line_acquisition="no (lab)", camera=UNK,
    line_scan="no", lighting="lab", manufacturer_count=UNK, fabric_types="8 texture classes", bboxes="derivatives only",
    masks="no", pixel_labels="no", metadata="class", annotation_quality="image-level only", known_limitations="old, lab, low-res",
    class_imbalance=UNK, potential_leakage="crops from same sample", train_test_protocol="none official", public="yes", download_size="small"),
 "AITEX": dict(year=2019, official_source="AITEX (Spanish textile institute) (verify URL)", license=UNK, image_count="245",
    normal_count="140", defective_count="105", defect_categories="12", resolution="4096x256", color="grayscale",
    static_or_sequential="static", continuous_production="no", production_line_acquisition="camera on loom (verify)",
    camera=UNK, line_scan=UNK, lighting=UNK, manufacturer_count="1", fabric_types="7", bboxes="derivable", masks="yes",
    pixel_labels="yes", metadata="fabric code in filename", annotation_quality="good but tiny", known_limitations="tiny; overfit risk",
    class_imbalance="moderate", potential_leakage="same fabric across splits", train_test_protocol="none official", public="yes", download_size="small"),
}

DS_MATRIX = pd.DataFrame(DATASETS).T
DS_MATRIX.to_csv(RESULTS_DIR / "dataset_matrix.csv")
DS_MATRIX[["year", "image_count", "resolution", "static_or_sequential", "continuous_production", "line_scan", "masks", "public"]]
''')

code(r'''# @title 03.2 Scoring framework (0–3 per criterion) with justifications
CRITERIA = ["industrial_realism", "continuous_production", "scale", "annotation_quality", "detection", "segmentation",
            "anomaly", "temporal", "cross_domain", "accessibility", "reproducibility", "research_potential"]

# (score, justification). Scores reflect the *current* evidence and must be revisited after acquisition/audit.
SCORES = {
 "TSFabrics": {
   "industrial_realism": (3, "captured on running knitting machines, varied speed/lighting"),
   "continuous_production": (3, "explicit time-series design"),
   "scale": (3, "~93k frames"),
   "annotation_quality": (2, "pixel-level, but quality not yet audited"),
   "detection": (2, "boxes derivable from masks"),
   "segmentation": (3, "pixel-level masks"),
   "anomaly": (2, "normal-heavy sequences usable for one-class"),
   "temporal": (3, "only candidate with true sequences"),
   "cross_domain": (2, "grayscale knitting only -> shift to woven data is large"),
   "accessibility": (1, "access/licence not yet confirmed"),
   "reproducibility": (2, "scenario metadata should allow group splits"),
   "research_potential": (3, "directly enables RQ3")},
 "RAW-FABRID": {
   "industrial_realism": (3, "line-scan + controlled LED, raw fabric"),
   "continuous_production": (1, "line-scan origin but static release"),
   "scale": (1, "709 images"),
   "annotation_quality": (2, "COCO + masks, not audited"),
   "detection": (3, "COCO boxes"), "segmentation": (3, "masks"), "anomaly": (2, "MVTec-style subset provided"),
   "temporal": (0, "none"), "cross_domain": (3, "different camera/fabric from TSFabrics"),
   "accessibility": (3, "Mendeley Data"), "reproducibility": (3, "CSV metadata + fixed split"),
   "research_potential": (2, "small-defect + segmentation + domain-shift target")},
 "Lusitano": {
   "industrial_realism": (3, "real company production, Linea line-scan"),
   "continuous_production": (2, "strips from continuous inspection; ordering to verify"),
   "scale": (3, "~32k normal images"), "annotation_quality": (1, "test masks — extent unknown"),
   "detection": (0, "no boxes"), "segmentation": (1, "test-only masks"), "anomaly": (3, "designed for one-class"),
   "temporal": (1, "possible if strip order recoverable"), "cross_domain": (3, "distinct domain"),
   "accessibility": (2, "public page; size may be heavy for Colab"), "reproducibility": (2, "MVTec layout"),
   "research_potential": (3, "primary anomaly benchmark for RQ2")},
 "Chenab Textile": {
   "industrial_realism": (2, "production line, but acquisition details unknown"), "continuous_production": (1, "static images"),
   "scale": (1, "unknown"), "annotation_quality": (1, "unknown"), "detection": (2, "YOLO studies exist"),
   "segmentation": (0, "none known"), "anomaly": (1, "unknown normal count"), "temporal": (0, "none"),
   "cross_domain": (3, "colour, printed fabric, different region"), "accessibility": (1, "shortened link, licence unclear"),
   "reproducibility": (1, "no official protocol"), "research_potential": (2, "external validation only")},
 "ZJU-Leaper": {
   "industrial_realism": (2, "assembly-line setting"), "continuous_production": (0, "static"), "scale": (3, "~98k"),
   "annotation_quality": (2, "boxes+masks, varies"), "detection": (3, ""), "segmentation": (3, ""), "anomaly": (2, "annotation-free track"),
   "temporal": (0, ""), "cross_domain": (3, "19 fabric groups enable within-dataset domain splits"),
   "accessibility": (1, "on request"), "reproducibility": (3, "official splits"), "research_potential": (2, "benchmark + generalisation")},
 "TILDA": {
   "industrial_realism": (1, "lab, 1996"), "continuous_production": (0, ""), "scale": (2, "3.2k"), "annotation_quality": (1, "image-level"),
   "detection": (1, "derivatives only"), "segmentation": (0, ""), "anomaly": (1, ""), "temporal": (0, ""),
   "cross_domain": (2, "old-domain stress test"), "accessibility": (3, ""), "reproducibility": (1, "no protocol"), "research_potential": (1, "literature comparison")},
 "AITEX": {
   "industrial_realism": (2, "loom camera, real fabrics"), "continuous_production": (0, ""), "scale": (0, "245"),
   "annotation_quality": (3, "clean masks"), "detection": (1, "tiny"), "segmentation": (2, "masks but tiny"), "anomaly": (1, "140 normals"),
   "temporal": (0, ""), "cross_domain": (2, "4096x256 strips resemble line-scan"), "accessibility": (3, ""),
   "reproducibility": (1, "no protocol; leakage-prone"), "research_potential": (1, "benchmark only")},
}

score_df = pd.DataFrame({d: {c: v[0] for c, v in s.items()} for d, s in SCORES.items()}).T[CRITERIA]
score_df["total"] = score_df.sum(axis=1)
score_df = score_df.sort_values("total", ascending=False)
score_df.to_csv(RESULTS_DIR / "dataset_scores.csv")

fig, ax = plt.subplots(figsize=(11, 4))
sns.heatmap(score_df[CRITERIA].astype(int), annot=True, cmap="Blues", cbar=False, ax=ax)
ax.set_title("Dataset suitability scores (0–3) — provisional, pre-audit")
savefig("03_dataset_scores.png")
score_df
''')

md(r"""### 03.3 Recommendation (provisional — revisited after audit in Section 05)

| Role | Dataset | Why |
|------|---------|-----|
| **Primary** | **TSFabrics** | Only candidate with true production sequences → RQ3 (temporal) and realistic continuous inspection. Pixel labels support detection + segmentation. |
| **Secondary (anomaly)** | **Lusitano** | Large normal-only training set from real production with a line-scan camera → RQ2 one-class setting. |
| **Secondary (detection/segmentation, small defects)** | **RAW-FABRID** | High-resolution line-scan, COCO + masks, immediately accessible → RQ5. |
| **External validation** | Chenab Textile, ZJU-Leaper (if access granted) | Different camera, colour, fabric and region → RQ4 domain shift. |
| **Benchmark / literature comparison** | AITEX, TILDA | Historical comparability only; too small / too old to be primary. |

**Uncertainties that could change this ranking:** TSFabrics access and licence; Lusitano download size on Colab; whether Lusitano strips preserve production order; Chenab provenance.

### Decision
Proceed to acquisition of **TSFabrics, RAW-FABRID, Lusitano** first. Acquire external-validation sets only after the primary pipeline exists.
""")
