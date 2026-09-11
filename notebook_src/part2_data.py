"""Sections 04-08: acquisition, audit, EDA, leakage, preparation."""

CELLS = []
md = lambda s: CELLS.append(("markdown", s))
code = lambda s: CELLS.append(("code", s))

md(r"""## 04. Dataset Acquisition

### Objective
Download **only** the justified datasets, reproducibly, and record provenance. Raw files are never modified.

### Method
Each dataset gets a `DatasetSpec` with an official source, an expected local layout, and a loader that returns a **unified manifest** (`image_path, mask_path, label, group, sequence_id, frame_idx, dataset`). The manifest is what every later section consumes — the code downstream never depends on dataset-specific folder layouts.

**Acquisition rule:** URLs for TSFabrics / Lusitano / Chenab must be pasted from the official pages at run time (they are not hard-coded here because they were not verified from the primary source). If a dataset requires a request form, place the archive in `RAW_DIR/<name>/` manually or via Drive.
""")

code(r'''# @title 04.1 Dataset registry + unified manifest schema
import hashlib, zipfile, tarfile, shutil, urllib.request
from dataclasses import dataclass, field

MANIFEST_COLUMNS = ["dataset", "image_path", "mask_path", "label", "defect_type", "group", "sequence_id", "frame_idx", "split_hint"]

@dataclass
class DatasetSpec:
    name: str
    official_source: str
    paper: str = UNK
    license: str = UNK
    download_urls: dict = field(default_factory=dict)   # {"archive.zip": "https://..."} — fill at run time
    notes: str = ""

    @property
    def raw_dir(self):
        return RAW_DIR / self.name

REGISTRY = {
    "TSFabrics": DatasetSpec("TSFabrics", official_source="<paste official TSFabrics page URL>",
                             notes="Sequential frames; keep scenario + frame order in manifest."),
    "RAW-FABRID": DatasetSpec("RAW-FABRID", official_source="Mendeley Data — <paste DOI URL>",
                              notes="Full-res images + COCO JSON + MVTec-style patches. Use full-res as RAW; patches are derived."),
    "Lusitano": DatasetSpec("Lusitano", official_source="https://kailashhambarde.github.io/Lusitano/",
                            notes="MVTec AD layout: train/good, test/good, test/<defect>, ground_truth/."),
    "Chenab Textile": DatasetSpec("Chenab Textile", official_source="<resolve shortened link to original host>"),
    "AITEX": DatasetSpec("AITEX", official_source="<AITEX official page>", notes="Defect_images/, NODefect_images/, Mask_images/"),
}

def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

def download(url, dest):
    dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print("exists:", dest.name); return dest
    print("downloading", url)
    urllib.request.urlretrieve(url, dest)
    return dest

def extract(archive, dest):
    archive, dest = Path(archive), Path(dest)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as z: z.extractall(dest)
    elif archive.suffix in {".tar", ".gz", ".tgz", ".bz2"}:
        with tarfile.open(archive) as t: t.extractall(dest)
    else:
        raise ValueError(f"unknown archive type: {archive}")

def acquire(spec: DatasetSpec):
    """Download + extract + write provenance. Idempotent; never touches existing extracted files."""
    spec.raw_dir.mkdir(parents=True, exist_ok=True)
    prov = {"dataset": spec.name, "source": spec.official_source, "paper": spec.paper, "license": spec.license,
            "download_date": datetime.datetime.now().isoformat(timespec="seconds"), "files": {}, "known_modifications": "none (RAW)"}
    for fname, url in spec.download_urls.items():
        arch = download(url, spec.raw_dir / "_archives" / fname)
        prov["files"][fname] = {"sha256": sha256_file(arch), "bytes": arch.stat().st_size}
        marker = spec.raw_dir / f".extracted_{fname}"
        if not marker.exists():
            extract(arch, spec.raw_dir); marker.touch()
    imgs = list_images(spec.raw_dir)
    prov["n_files"] = sum(1 for _ in spec.raw_dir.rglob("*") if _.is_file())
    prov["n_images"] = len(imgs)
    save_json(prov, f"provenance_{spec.name.replace(' ', '_')}.json")
    print(f"{spec.name}: {prov['n_images']} images under {spec.raw_dir}")
    return prov

if RUN["ACQUIRE_DATA"]:
    for name in ["RAW-FABRID", "Lusitano", "TSFabrics"]:
        spec = REGISTRY[name]
        if not spec.download_urls and not list_images(spec.raw_dir):
            print(f"[{name}] no download_urls set and no files present — paste URLs from {spec.official_source} into REGISTRY[...].download_urls")
        else:
            acquire(spec)
''')

code(r'''# @title 04.2 Manifest builders (one per dataset layout) -> unified manifest
def _label_from_dir(parts, good_tokens=("good", "normal", "nodefect", "ok")):
    joined = "/".join(parts).lower()
    return 0 if any(t in joined for t in good_tokens) else 1

def manifest_mvtec_layout(spec, dataset_name=None):
    """MVTec AD layout: <root>/train/good, <root>/test/<class>, <root>/ground_truth/<class>/<stem>_mask.png"""
    root = spec.raw_dir; rows = []
    for img in list_images(root):
        rel = img.relative_to(root).parts
        if "ground_truth" in rel or rel[0].startswith("_"):
            continue
        split_hint = "train" if "train" in rel else ("test" if "test" in rel else UNK)
        defect_type = rel[-2] if len(rel) >= 2 else UNK
        label = 0 if defect_type.lower() == "good" else 1
        mask = None
        gt_dir = root / "ground_truth" / defect_type
        for cand in [gt_dir / f"{img.stem}_mask.png", gt_dir / f"{img.stem}.png"]:
            if cand.exists(): mask = cand; break
        rows.append(dict(dataset=dataset_name or spec.name, image_path=str(img), mask_path=str(mask) if mask else None,
                         label=label, defect_type=defect_type if label else "none", group=UNK,
                         sequence_id=UNK, frame_idx=np.nan, split_hint=split_hint))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

def manifest_aitex(spec):
    root = spec.raw_dir; rows = []
    for img in list_images(root):
        low = str(img).lower()
        if "mask" in low: continue
        label = 0 if "nodefect" in low else 1
        # AITEX filename convention: <defectcode>_<fabriccode>_<id>.png  (verify on actual files)
        parts = img.stem.split("_")
        fabric = parts[1] if len(parts) > 1 else UNK
        mask = next((m for m in root.rglob(f"{img.stem}_mask.png")), None)
        rows.append(dict(dataset="AITEX", image_path=str(img), mask_path=str(mask) if mask else None, label=label,
                         defect_type=parts[0] if label else "none", group=f"fabric_{fabric}", sequence_id=UNK,
                         frame_idx=np.nan, split_hint=UNK))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

def manifest_sequential_generic(spec, seq_depth=1):
    """For TSFabrics-like layouts: <root>/<scenario>/<...>/<frame_0001>.png with masks in a parallel folder.
    Frame index is parsed from trailing digits. Adjust after inspecting the real folder structure."""
    import re
    root = spec.raw_dir; rows = []
    for img in list_images(root):
        rel = img.relative_to(root).parts
        if rel[0].startswith("_") or any(k in str(img).lower() for k in ["mask", "label", "annotation"]):
            continue
        scenario = "/".join(rel[:seq_depth])
        m = re.search(r"(\d+)$", img.stem)
        frame_idx = int(m.group(1)) if m else np.nan
        mask = None
        for cand in root.rglob(f"{img.stem}*"):
            if cand != img and cand.suffix.lower() in IMG_EXTS and "mask" in str(cand).lower():
                mask = cand; break
        label = np.nan
        if mask is not None:
            label = int(read_image(mask, gray=True).max() > 0)
        rows.append(dict(dataset=spec.name, image_path=str(img), mask_path=str(mask) if mask else None, label=label,
                         defect_type=UNK, group=scenario, sequence_id=scenario, frame_idx=frame_idx, split_hint=UNK))
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)

def manifest_coco(spec, json_glob="**/*.json"):
    """RAW-FABRID-like: full-res images + COCO annotations. Label = has >=1 annotation."""
    root = spec.raw_dir
    jsons = [p for p in root.glob(json_glob) if "patch" not in str(p).lower()]
    if not jsons:
        print("No COCO json found under", root); return pd.DataFrame(columns=MANIFEST_COLUMNS)
    coco = json.loads(jsons[0].read_text())
    img_by_id = {im["id"]: im for im in coco["images"]}
    anns = {}
    for a in coco["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    cats = {c["id"]: c["name"] for c in coco.get("categories", [])}
    rows = []
    for iid, im in img_by_id.items():
        path = next((p for p in root.rglob(Path(im["file_name"]).name)), None)
        if path is None: continue
        a = anns.get(iid, [])
        rows.append(dict(dataset=spec.name, image_path=str(path), mask_path=None, label=int(len(a) > 0),
                         defect_type=",".join(sorted({cats.get(x["category_id"], str(x["category_id"])) for x in a})) or "none",
                         group=UNK, sequence_id=UNK, frame_idx=np.nan, split_hint=UNK))
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    df.attrs["coco_json"] = str(jsons[0])
    return df

MANIFEST_BUILDERS = {
    "Lusitano": lambda s: manifest_mvtec_layout(s),
    "AITEX": manifest_aitex,
    "TSFabrics": lambda s: manifest_sequential_generic(s, seq_depth=1),
    "RAW-FABRID": manifest_coco,
    "Chenab Textile": lambda s: manifest_mvtec_layout(s),   # replace once real layout is inspected
}

def build_manifest(name, force=False):
    out = PROCESSED_DIR / f"manifest_{name.replace(' ', '_')}.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    spec = REGISTRY[name]
    if not list_images(spec.raw_dir):
        print(f"[{name}] no images in {spec.raw_dir}"); return pd.DataFrame(columns=MANIFEST_COLUMNS)
    df = MANIFEST_BUILDERS[name](spec)
    df.to_csv(out, index=False)
    print(f"[{name}] manifest: {len(df)} rows, {df['label'].sum() if df['label'].notna().any() else 'n/a'} defective")
    return df

MANIFESTS = {}
if RUN["ACQUIRE_DATA"]:
    for name in ["RAW-FABRID", "Lusitano", "TSFabrics"]:
        MANIFESTS[name] = build_manifest(name)
''')

md(r"""## 05. Dataset Audit & Provenance

### Objective
Establish that the data is what the papers say it is before training anything. Every statistic here has a purpose:

| Check | Purpose |
|-------|---------|
| unreadable / corrupted files | training crashes, silent sample loss |
| exact duplicates (SHA-256) | leakage, inflated metrics |
| near duplicates (perceptual hash) | leakage from consecutive frames / crops |
| dimensions, channels, aspect ratio | preprocessing decisions (resize vs tile) |
| mask validity + image/mask correspondence | segmentation labels that are actually wrong |
| class and defect-area distribution | imbalance strategy, small-defect analysis |
| brightness / contrast | illumination variation across scenarios → domain shift indicator |
""")

code(r'''# @title 05.1 Audit functions
import imagehash
from PIL import Image

def audit_images(manifest, sample=None, phash_size=8):
    """Per-image stats. `sample` limits the number of rows for very large datasets (documented in results)."""
    df = manifest if sample is None else manifest.sample(min(sample, len(manifest)), random_state=SEED)
    recs = []
    for r in df.itertuples():
        rec = dict(image_path=r.image_path, readable=True)
        try:
            img = read_image(r.image_path)
            rec.update(h=img.shape[0], w=img.shape[1], c=img.shape[2] if img.ndim == 3 else 1,
                       mean=float(img.mean()), std=float(img.std()),
                       sha256=sha256_file(r.image_path),
                       phash=str(imagehash.phash(Image.fromarray(img), hash_size=phash_size)))
            if isinstance(r.mask_path, str):
                m = read_image(r.mask_path, gray=True)
                rec.update(mask_ok=m.shape[:2] == img.shape[:2], mask_area_ratio=float((m > 0).mean()),
                           mask_unique=int(len(np.unique(m))))
        except Exception as e:
            rec.update(readable=False, error=str(e))
        recs.append(rec)
    out = pd.DataFrame(recs)
    out["aspect"] = out["w"] / out["h"]
    return out

def find_duplicates(audit_df, phash_thresh=4):
    exact = audit_df[audit_df.duplicated("sha256", keep=False)].sort_values("sha256")
    hashes = [(p, imagehash.hex_to_hash(h)) for p, h in zip(audit_df.image_path, audit_df.phash) if isinstance(h, str)]
    near = []
    # O(n^2) is fine up to a few thousand images; use FAISS/bucketing for larger sets
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            d = hashes[i][1] - hashes[j][1]
            if d <= phash_thresh:
                near.append((hashes[i][0], hashes[j][0], d))
    near = pd.DataFrame(near, columns=["a", "b", "hamming"])
    return exact, near

def audit_report(name, manifest, sample=None):
    a = audit_images(manifest, sample=sample)
    exact, near = find_duplicates(a)
    rep = {
        "dataset": name, "n_audited": len(a), "n_unreadable": int((~a.readable).sum()),
        "dims": a.groupby(["h", "w", "c"]).size().to_dict(),
        "n_exact_dup_files": len(exact), "n_near_dup_pairs": len(near),
        "brightness_mean": float(a["mean"].mean()), "brightness_std_across_images": float(a["mean"].std()),
        "mask_shape_mismatch": int((a.get("mask_ok", pd.Series(dtype=bool)) == False).sum()),
    }
    a.to_csv(RESULTS_DIR / f"audit_{name.replace(' ', '_')}.csv", index=False)
    near.to_csv(RESULTS_DIR / f"neardups_{name.replace(' ', '_')}.csv", index=False)
    save_json({k: (str(v) if isinstance(v, dict) else v) for k, v in rep.items()}, f"audit_{name.replace(' ', '_')}.json")
    print(json.dumps({k: v for k, v in rep.items() if k != "dims"}, indent=2))
    return a, exact, near

AUDITS = {}
if RUN["AUDIT"]:
    for name, mf in MANIFESTS.items():
        if len(mf):
            AUDITS[name] = audit_report(name, mf, sample=3000 if len(mf) > 3000 else None)
''')

md(r"""### 05.2 Interpretation template
After running the audit, fill in:

* **Readable**: any unreadable files? → drop and record in provenance `known_modifications`.
* **Duplicates**: exact duplicates must be removed before any split; near-duplicate pairs feed Section 07.
* **Dimensions**: a single fixed size (e.g. 800×640) means resize is safe; strip shapes (4096×1024) mean **tiling is mandatory** for CNN input.
* **Masks**: any shape mismatch → annotation error; `mask_unique > 2` → multi-class or anti-aliased mask (binarise deliberately).
* **Brightness spread**: high std across images within one dataset suggests illumination variation — a candidate domain-shift factor.
""")

md(r"""## 06. Exploratory Data Analysis

### Objective
Answer concrete questions, not decorate: (1) class balance, (2) defect size relative to image, (3) illumination variation per group/scenario, (4) what defects actually look like.
""")

code(r'''# @title 06.1 EDA plots
def eda(name, manifest, audit_df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    if manifest["label"].notna().any():
        manifest["label"].value_counts().rename({0: "normal", 1: "defective"}).plot.bar(ax=axes[0], color=["#4C72B0", "#C44E52"])
        axes[0].set_title("Class distribution")
    if "mask_area_ratio" in audit_df and audit_df["mask_area_ratio"].notna().any():
        ar = audit_df["mask_area_ratio"].dropna(); ar = ar[ar > 0]
        axes[1].hist(np.log10(ar + 1e-6), bins=40, color="#55A868")
        axes[1].set_title("log10(defect area / image area), defective only"); axes[1].set_xlabel("log10 ratio")
        print(f"defect area ratio: median={ar.median():.4%}, p10={ar.quantile(.1):.4%}, p90={ar.quantile(.9):.4%}")
    merged = manifest.merge(audit_df[["image_path", "mean"]], on="image_path", how="inner")
    if merged["group"].nunique() > 1 and merged["group"].nunique() < 60:
        sns.boxplot(data=merged, x="group", y="mean", ax=axes[2]); axes[2].tick_params(axis="x", rotation=90)
        axes[2].set_title("Brightness by group / scenario")
    else:
        axes[2].hist(merged["mean"], bins=40); axes[2].set_title("Brightness distribution")
    fig.suptitle(f"EDA — {name}"); plt.tight_layout(); savefig(f"06_eda_{name.replace(' ', '_')}.png")

    if manifest["label"].notna().any():
        defect = manifest[manifest.label == 1].sample(min(8, int((manifest.label == 1).sum())), random_state=SEED)
        masks = [m if isinstance(m, str) else None for m in defect.mask_path]
        show_grid(list(defect.image_path), titles=list(defect.defect_type), masks=masks, gray=True)
        plt.suptitle(f"{name}: defective samples with masks"); savefig(f"06_samples_{name.replace(' ', '_')}.png")

if RUN["EDA"]:
    for name in AUDITS:
        eda(name, MANIFESTS[name], AUDITS[name][0])
''')

md(r"""## 07. Data Leakage Analysis

### Objective
Treat leakage as a first-class research problem. A random image-level split on sequential or patch data produces **optimistically wrong** results.

### Leakage types checked
| Type | Detection | Mitigation |
|------|-----------|------------|
| Exact duplicate | SHA-256 | remove |
| Near duplicate | pHash Hamming ≤ 4 | assign both to same split |
| Sequential-frame | consecutive `frame_idx` in same `sequence_id` | **group split by sequence** |
| Same roll / batch / fabric | `group` column | **group split** |
| Patch ↔ parent image | patch provenance | split at parent level, then tile |
| Mask/image mismatch | audit `mask_ok` | drop / fix |

### Method
`GroupShuffleSplit` on the leakage-safe group key. If no group metadata exists, near-duplicate connected components become the groups. The split is saved as a manifest CSV in `SPLITS_DIR` and never regenerated silently.
""")

code(r'''# @title 07.1 Leakage-safe splitting
from sklearn.model_selection import GroupShuffleSplit
import networkx as nx

def near_dup_groups(manifest, near_df):
    """Connected components over near-duplicate edges become group ids."""
    G = nx.Graph(); G.add_nodes_from(manifest.image_path)
    G.add_edges_from(zip(near_df.a, near_df.b))
    comp = {}
    for i, c in enumerate(nx.connected_components(G)):
        for p in c: comp[p] = f"dup_{i}"
    return manifest.image_path.map(comp)

def make_group_split(name, manifest, near_df=None, test_size=0.2, val_size=0.1, force=False):
    out = SPLITS_DIR / f"split_{name.replace(' ', '_')}.csv"
    if out.exists() and not force:
        return pd.read_csv(out)
    df = manifest.copy()
    # Priority of group key: explicit sequence/group metadata > near-dup component > image itself
    key = df["sequence_id"].where(df["sequence_id"] != UNK, df["group"])
    if (key == UNK).all() and near_df is not None and len(near_df):
        key = near_dup_groups(df, near_df).fillna(df.image_path)
        strategy = "near-duplicate components"
    elif (key == UNK).all():
        key = df.image_path; strategy = "image-level (no group metadata found — leakage risk documented)"
    else:
        strategy = "group metadata (sequence/group)"
    df["split_group"] = key
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    tr_idx, te_idx = next(gss.split(df, groups=df.split_group))
    df.loc[df.index[te_idx], "split"] = "test"
    train_part = df.iloc[tr_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size / (1 - test_size), random_state=SEED)
    tr2, va2 = next(gss2.split(train_part, groups=train_part.split_group))
    df.loc[train_part.index[va2], "split"] = "val"
    df.loc[train_part.index[tr2], "split"] = "train"
    # Respect official train split for one-class datasets: official train must stay normal-only
    if (df.split_hint == "train").any():
        df.loc[df.split_hint == "train", "split"] = "train"
    df.to_csv(out, index=False)
    overlap = set(df[df.split == "train"].split_group) & set(df[df.split == "test"].split_group)
    print(f"[{name}] strategy={strategy}; groups train/test overlap={len(overlap)}")
    print(df.groupby(["split", "label"]).size().unstack(fill_value=0))
    return df

SPLITS = {}
if RUN["LEAKAGE"]:
    for name in MANIFESTS:
        near = AUDITS[name][2] if name in AUDITS else None
        SPLITS[name] = make_group_split(name, MANIFESTS[name], near)
''')

md(r"""### 07.2 Decision record
Document here after running:

* **Split key used per dataset** and why (e.g. TSFabrics → scenario/sequence; Lusitano → official train + near-dup groups; RAW-FABRID → parent image).
* **Frames/images dropped** as exact duplicates.
* **Residual risk**: e.g. two scenarios recorded on the same fabric roll cannot be separated without metadata → stated as a limitation.
""")

md(r"""## 08. Dataset Preparation

### Objective
Reusable, non-destructive preprocessing. RAW is never overwritten; tiles and resized copies go to `PROCESSED_DIR` with a deterministic naming scheme so they can be regenerated.

### Full image vs patches — an experiment, not an assumption
Line-scan strips (4096×1024, 1792×1024) cannot be fed at native resolution; resizing a 4096-wide strip to 256 px destroys defects that are ~20 px wide. We therefore support **both** paths and compare them in Section 19:

* `resize` — whole image → `IMG_SIZE` (fast, loses small defects)
* `tile` — non-overlapping (or overlapping) tiles at native resolution; tile label = mask has defect pixels (or box overlap)

Patch leakage is prevented because tiling happens **after** the group split (tiles inherit the parent's split).
""")

code(r'''# @title 08.1 Tiling + torch datasets
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.v2 as T

def tile_manifest(split_df, tile=512, stride=None, min_defect_px=20, out_root=None, force=False):
    """Create tiles from full images. Returns a tile-level manifest inheriting parent split/group."""
    stride = stride or tile
    out_root = Path(out_root or PROCESSED_DIR / f"tiles_{tile}")
    rows = []
    for r in split_df.itertuples():
        img = read_image(r.image_path, gray=True)
        mask = read_image(r.mask_path, gray=True) if isinstance(r.mask_path, str) else None
        H, W = img.shape
        for y in range(0, max(H - tile, 0) + 1, stride):
            for x in range(0, max(W - tile, 0) + 1, stride):
                stem = f"{Path(r.image_path).stem}_y{y}_x{x}"
                ip = out_root / r.dataset / "images" / f"{stem}.png"
                mp = out_root / r.dataset / "masks" / f"{stem}.png" if mask is not None else None
                if force or not ip.exists():
                    ip.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(ip), img[y:y + tile, x:x + tile])
                    if mask is not None:
                        mp.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(mp), mask[y:y + tile, x:x + tile])
                if mask is not None:
                    label = int((mask[y:y + tile, x:x + tile] > 0).sum() >= min_defect_px)
                else:
                    label = r.label if pd.notna(r.label) else np.nan
                rows.append(dict(dataset=r.dataset, image_path=str(ip), mask_path=str(mp) if mp else None, label=label,
                                 defect_type=r.defect_type if label == 1 else "none", group=r.group, sequence_id=r.sequence_id,
                                 frame_idx=r.frame_idx, split=r.split, parent=r.image_path, tile_y=y, tile_x=x))
    df = pd.DataFrame(rows)
    df.to_csv(SPLITS_DIR / f"tiles_{tile}_{split_df.dataset.iloc[0].replace(' ', '_')}.csv", index=False)
    print(f"tiles: {len(df)} | defective {int(df.label.sum()) if df.label.notna().any() else 'n/a'}")
    return df

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

def build_transforms(size=IMG_SIZE, train=True, augment=True):
    ops = [T.ToImage(), T.Resize((size, size), antialias=True)]
    if train and augment:
        # Fabric is roughly translation/flip invariant; avoid colour jitter on grayscale
        ops += [T.RandomHorizontalFlip(), T.RandomVerticalFlip(), T.RandomAffine(degrees=0, translate=(0.05, 0.05))]
    ops += [T.ToDtype(torch.float32, scale=True), T.Normalize(MEAN, STD)]
    return T.Compose(ops)

class FabricDataset(Dataset):
    def __init__(self, df, transform=None, with_mask=False, size=IMG_SIZE):
        self.df = df.reset_index(drop=True); self.tf = transform or build_transforms(size, train=False)
        self.with_mask = with_mask; self.size = size
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = read_image(r.image_path)          # 3-channel so ImageNet backbones work on grayscale sources too
        x = self.tf(img)
        y = torch.tensor(int(r.label) if pd.notna(r.label) else -1)
        if not self.with_mask:
            return x, y
        if isinstance(r.mask_path, str):
            m = read_image(r.mask_path, gray=True)
        else:
            m = np.zeros(img.shape[:2], np.uint8)
        m = cv2.resize(m, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        return x, torch.from_numpy((m > 0).astype(np.float32))[None], y

def make_loader(df, train, size=IMG_SIZE, batch=BATCH_SIZE, with_mask=False, augment=True):
    ds = FabricDataset(df, build_transforms(size, train, augment), with_mask, size)
    return DataLoader(ds, batch_size=batch, shuffle=train, num_workers=NUM_WORKERS, pin_memory=True, drop_last=train)

if RUN["PREPARE"]:
    TILES = {name: tile_manifest(SPLITS[name], tile=512) for name in SPLITS if len(SPLITS[name])}
''')
