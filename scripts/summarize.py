#!/usr/bin/env python3
"""Collect runs/*/*/val_final.json into a Markdown table."""
import json
from pathlib import Path

rows = []
for f in sorted(Path("runs").glob("*/*/val_final.json")):
    v = json.loads(f.read_text()); m = v["metrics"]
    rows.append(f"| {f.parts[1]} | {f.parts[2]} | {v['params_M']} | {m['iou']:.3f} | {m['dice']:.3f} | "
                f"{m['precision']:.3f} | {m['recall']:.3f} | {v['best_epoch']} | {v['train_minutes']} |")
print("| class | arch | params (M) | IoU | Dice | precision | recall | best epoch | minutes |")
print("|---|---|---|---|---|---|---|---|---|")
print("\n".join(rows) if rows else "(no runs found)")
