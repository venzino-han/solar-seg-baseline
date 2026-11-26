#!/usr/bin/env python3
"""Build the per-class release: original image + box XML + final segmentation label for every box.

Label per box = human label if one exists (label_source=human), else the confirmed mask of the final fine-tuned
SAM 3 label pack (label_source=sam3), else the regenerated mask of the retry (label_source=sam3_retry), else a brightness-rule mask (label_source=rule). Labeller identity is not exported.

    python3 build_release.py [--out release_20261126]
"""
import argparse, csv, json, re, shutil, sys, zipfile
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
import cv2, numpy as np
B = Path(__file__).resolve().parent; DATA = B.parent / "dataset"
CLS = dict(coronal_hole=dict(pack="label_pack", chan="AIA_193", score="confidence"),
           sunspot=dict(pack="label_pack_sunspot_ft", chan="HMI_Ic", score="sam_cv_iou"),
           prominence=dict(pack="label_pack_prominence_hs133", chan="AIA_304", score="sam_cv_iou"))


def human(cls):
    """{(stem, bi): (mask png path, edited 0/1)}; later sources override earlier ones (as in training)."""
    H = {}
    if cls == "coronal_hole":
        for l in open(B / "gold/gold.jsonl"):
            x = json.loads(l); H[(x["stem"], x["bi"])] = (B / "gold/masks_512" / f"{x['sid']}.png", int(x["source"] == "edited"))
    else:
        for g in dict(sunspot=["gold_sunspot"], prominence=["gold_prominence", "gold_prominence_kasi", "gold_prominence_hs"])[cls]:
            for r in csv.DictReader(open(B / g / "labels.csv")):
                H[(r["stem"], int(r["bi"]))] = (B / g / r["mask_path"], int(r["edited"]))
    return H


def retried(cls):
    """{(stem, bi): mask png} for boxes whose first mask failed the quality gate and that were regenerated (retry_select.py / retry_ch_vote.py)."""
    O = B / "retry_unlabeled" / cls
    if cls == "coronal_hole":
        S = json.load(open(O / "summary.json")); return {(S["stem"], S["box_index"]): O / "majority_mask.png"}
    D = {(p.stem.rsplit("__b", 1)[0], int(p.stem.rsplit("__b", 1)[1])): p for p in (O / "masks").glob("*.png")}
    if (O / "fallback.csv").exists():                                # last resort (retry_fallback.py): any gate-passing variant, else brightness rule
        for r in csv.DictReader(open(O / "fallback.csv")):
            D.setdefault((r["stem"], int(r["bi"])), (O / "masks_fallback" / f"{r['stem']}__b{r['bi']}.png", r["tier"]))
    return D


def build(cls, out):
    c = CLS[cls]; P = B / c["pack"]; R = out / f"solar_{cls}_seg"; H = human(cls); used = set(); RT = retried(cls)
    if R.exists(): shutil.rmtree(R)
    rows = list(csv.DictReader(open(P / "labels.csv"))); byimg = defaultdict(list)
    for r in rows: byimg[(r["split"], r["stem"])].append(r)
    imgs = sorted((s, p.stem) for s in ("train", "test") for p in (DATA / s).glob(f"*_{c['chan']}_512.jpg"))
    for s in ("train", "test"):
        for d in ("images", "boxes", "masks", "masks_full"): (R / d / s).mkdir(parents=True)
    table, st = [], Counter()
    for split, stem in imgs:
        shutil.copy(DATA / split / f"{stem}.jpg", R / "images" / split / f"{stem}.jpg")
        xml = (DATA / split / f"{stem}.xml").read_text(encoding="utf-8", errors="replace")
        (R / "boxes" / split / f"{stem}.xml").write_text(re.sub(r"<path>.*?</path>", f"<path>{stem}.jpg</path>", xml, flags=re.S), encoding="utf-8")
        xb = [[int(float(o.find("bndbox").find(k).text)) for k in ("xmin", "ymin", "xmax", "ymax")] for o in ET.fromstring(xml).iter("object") if o.find("name").text == cls]
        rs = sorted(byimg.get((split, stem), []), key=lambda r: int(r["bi"]))
        assert [[int(v) for v in r["box"].split()] for r in rs] == xb, (stem, xb)
        full = np.zeros((512, 512), bool); st["images"] += 1; st[f"images_{split}"] += 1
        for r in rs:
            bi = int(r["bi"]); k = (stem, bi); src, ed, m = "none", "", None
            if k in H:
                m = cv2.imread(str(H[k][0]), 0); assert m is not None and m.shape == (512, 512), H[k][0]
                m = m > 127; src, ed = "human", H[k][1]; used.add(k); assert split == "test"
            elif r["confirmed"] == "1":
                m = cv2.imread(str(P / r["mask_path"]), 0); assert m is not None and m.shape == (512, 512), r["mask_path"]
                m = m > 127; src = "sam3"
            elif k in RT:
                rp, src = RT[k] if isinstance(RT[k], tuple) else (RT[k], "sam3_retry")
                m = cv2.imread(str(rp), 0); assert m is not None and m.shape == (512, 512) and m.any(), rp
                m = m > 127
            x1, y1, x2, y2 = xb[bi]; mp = ""
            if m is not None:
                mp = f"masks/{split}/{stem}__b{bi}.png"; cv2.imwrite(str(R / mp), m.astype(np.uint8) * 255); full |= m
            st[f"{src}_{split}"] += 1; st[src] += 1
            table.append(dict(split=split, stem=stem, box_index=bi, xmin=x1, ymin=y1, xmax=x2, ymax=y2, label_source=src, human_edited=ed,
                              area_px=int(m.sum()) if m is not None else "", model_score=r[c["score"]] if src == "sam3" else "",
                              image_path=f"images/{split}/{stem}.jpg", box_path=f"boxes/{split}/{stem}.xml", mask_path=mp, mask_full_path=f"masks_full/{split}/{stem}.png"))
        cv2.imwrite(str(R / "masks_full" / split / f"{stem}.png"), full.astype(np.uint8) * 255)
        if not rs: st["images_without_box"] += 1
        elif not full.any(): st["images_without_label"] += 1
    assert used == set(H), set(H) - used
    assert len(table) == len(rows)
    with open(R / "labels.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(table[0])); w.writeheader(); w.writerows(table)
    for src in ("human", "sam3", "sam3_retry", "rule"):
        a = [t["area_px"] for t in table if t["label_source"] == src]; st[f"{src}_area_median"] = int(np.median(a)) if a else 0
    st["human_edited"] = sum(t["human_edited"] == 1 for t in table); st["boxes"] = len(table)
    return R, dict(st)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="release_20261126"); ap.add_argument("--zip", action="store_true"); a = ap.parse_args()
    out = B / a.out; out.mkdir(exist_ok=True); stats = {}
    if not a.zip:
        for cls in CLS:
            R, stats[cls] = build(cls, out); print(cls, stats[cls], flush=True)
        (out / "stats.json").write_text(json.dumps(stats, indent=1))
    else:
        for cls in CLS:
            R = out / f"solar_{cls}_seg"; shutil.copy(out / "DATASET.md", R / "DATASET.md"); z = out / f"{R.name}.zip"
            with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
                for p in sorted(R.rglob("*")):
                    if p.is_file(): zf.write(p, p.relative_to(out))
            print(z.name, f"{z.stat().st_size/1e6:.0f} MB", len(zipfile.ZipFile(z).namelist()), "files", flush=True)
