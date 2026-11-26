#!/usr/bin/env python3
"""Rule-based completeness check of a release folder (no visual inspection).

Per class it checks, for every image and every box of the original XML:
  files   image / box XML / full mask exist; one labels.csv row per XML box, same coordinates
  label   a mask file exists for the box, is 512x512 binary, and has at least MIN_PX labelled pixels
  place   the mask has labelled pixels inside its box (human labels may extend outside)
  union   masks_full equals the union of the image's box masks
Writes <release>/verify_report.json and <release>/verify_problems.csv; exit code 1 if any file-level rule fails.

    python3 verify_release.py [--out release_20261126] [--zip]     (--zip also checks the zip archives against the folders)
"""
import argparse, csv, json, sys, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent; DATA = B.parent / "dataset"
CHAN = dict(coronal_hole="AIA_193", sunspot="HMI_Ic", prominence="AIA_304")
MIN_PX = dict(coronal_hole=60, sunspot=4, prominence=8)          # the pipeline's own minimum area per class


def check(cls, out):
    R = out / f"solar_{cls}_seg"; rows = list(csv.DictReader(open(R / "labels.csv", encoding="utf-8"))); by = defaultdict(list); P = []; C = Counter()
    bad = lambda r, rule, detail="": (P.append(dict(cls=cls, split=r["split"], stem=r["stem"], box_index=r.get("box_index", ""), label_source=r.get("label_source", ""), rule=rule, detail=detail)), C.update([rule]))
    for r in rows: by[(r["split"], r["stem"])].append(r)
    src = sorted((s, p.stem) for s in ("train", "test") for p in (DATA / s).glob(f"*_{CHAN[cls]}_512.jpg")); C["images"] = len(src); C["boxes"] = len(rows)
    for split, stem in src:
        r0 = dict(split=split, stem=stem)
        for d, e in (("images", "jpg"), ("boxes", "xml"), ("masks_full", "png")):
            if not (R / d / split / f"{stem}.{e}").is_file(): bad(r0, f"missing_{d}_file")
        xb = [[int(float(o.find("bndbox").find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")] for o in ET.parse(DATA / split / f"{stem}.xml").getroot().iter("object") if o.find("name").text == cls]
        rs = sorted(by.pop((split, stem), []), key=lambda r: int(r["box_index"]))
        if [[int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax")] for r in rs] != xb: bad(r0, "rows_do_not_match_xml_boxes", f"{len(rs)} rows vs {len(xb)} boxes"); continue
        union = np.zeros((512, 512), bool)
        for r in rs:
            C[f"source_{r['label_source']}"] += 1
            if r["label_source"] == "none" or not r["mask_path"]: bad(r, "no_label"); continue
            m = cv2.imread(str(R / r["mask_path"]), cv2.IMREAD_UNCHANGED)
            if m is None: bad(r, "missing_mask_file", r["mask_path"]); continue
            if m.shape != (512, 512) or not set(np.unique(m)) <= {0, 255}: bad(r, "mask_not_512_binary", str(m.shape)); continue
            m = m > 0; n = int(m.sum()); union |= m; x1, y1, x2, y2 = (int(r[k]) for k in ("xmin", "ymin", "xmax", "ymax"))
            if n != int(r["area_px"]): bad(r, "area_differs_from_csv", f"{n} vs {r['area_px']}")
            if n < MIN_PX[cls]: bad(r, "too_few_pixels", f"{n} < {MIN_PX[cls]}")
            if not m[y1:y2, x1:x2].any(): bad(r, "no_pixels_inside_box")
            C["labelled"] += 1
        f = cv2.imread(str(R / "masks_full" / split / f"{stem}.png"), 0)
        if f is not None and not np.array_equal(f > 0, union): bad(r0, "full_mask_is_not_union")
    for (split, stem), rs in by.items(): bad(dict(split=split, stem=stem), "rows_without_source_image", str(len(rs)))
    return dict(C), P


def check_zip(cls, out):
    R = out / f"solar_{cls}_seg"; z = zipfile.ZipFile(out / f"{R.name}.zip"); names = {i.filename: i.file_size for i in z.infolist()}
    files = {str(p.relative_to(out)): p.stat().st_size for p in R.rglob("*") if p.is_file()}
    return dict(zip_files=len(names), folder_files=len(files), same_names_and_sizes=names == files, crc_ok=z.testzip() is None)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="release_20261126"); ap.add_argument("--zip", action="store_true"); a = ap.parse_args(); out = B / a.out; rep, prob = {}, []
    for cls in CHAN:
        rep[cls], p = check(cls, out); prob += p
        if a.zip: rep[cls]["zip"] = check_zip(cls, out)
        print(cls, json.dumps(rep[cls], ensure_ascii=False), flush=True)
    (out / "verify_report.json").write_text(json.dumps(rep, indent=1))
    with open(out / "verify_problems.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["cls", "split", "stem", "box_index", "label_source", "rule", "detail"]); w.writeheader(); w.writerows(prob)
    hard = [p for p in prob if p["rule"] not in ("no_label", "too_few_pixels", "no_pixels_inside_box")]
    sys.exit(1 if hard or any(not all(v for k, v in r.get("zip", {}).items() if k in ("same_names_and_sizes", "crc_ok")) for r in rep.values()) else 0)
