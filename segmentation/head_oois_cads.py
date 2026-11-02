#!/usr/bin/env python3
"""
head_oois_cads.py -- the head organs of interest the whole-body models do not give, from CADS (optional step 1c).

CADS model 558 (Xu et al., arXiv:2507.22953; https://github.com/murong-xu/CADS) is a CT-only head-and-neck
nnU-Net v2. This step takes from it the brainstem, cochleae, chiasm, pituitary and lacrimal glands of a TMI
head-first scan:
  1. crop the CT from the brain down to the thyroid (TotalSegmentator brain, skull and thyroid_gland, +15 mm): a
     head-only crop leaves out the neck the model expects;
  2. write the crop in RAS voxel order, as CADS' own preprocessing does. nnUNetv2_predict does not reorient, and on
     a crop in DICOM (LPS) order the model returns next to nothing;
  3. run nnUNetv2_predict on it (a CPU is enough: minutes per crop);
  4. put the labels back on the CT grid the models saw and write one mask per structure, plus a CSV with the
     volume and a left/right check against the brainstem.

Masks only: build_case.py does not merge them into the RTSTRUCT. They are unreviewed automatic contours.

Weights: Dataset558_OAR258.zip from the CADS releases, unzipped into <weights>/ ("research": CC BY-NC-SA 4.0,
"open": CC BY-SA 4.0), e.g.
  https://github.com/murong-xu/CADS/releases/download/cads-model-research_v1.0.0/Dataset558_OAR258.zip

  python segmentation/head_oois_cads.py --nifti <work>/nifti/P001/20260101/CT_2.nii.gz \\
      --ts <work>/segmentations/totalsegmentator/total/P001_20260101_CT_2 \\
      --weights <weights> --out <output folder> [--device cuda]
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage

# CADS model 558 label values (CADS resources/info/labelmap.md) -> structure names as in tmli_v2.NNI_NAMES
LABELS = {"BrainStem": 5, "Cochlea_L": 8, "Cochlea_R": 9, "LacrimalGland_L": 16, "LacrimalGland_R": 17,
          "Chiasm": 24, "Pituitary": 29}
CROP_FROM = ("brain", "skull", "thyroid_gland")      # TotalSegmentator 'total' classes that span the crop
MARGIN_MM = 15.0
NNUNET = dict(d="558", c="3d_fullres", f="all", tr="nnUNetTrainerNoMirroring", p="nnUNetResEncUNetLPlans")


def largest(m):
    lab, n = ndimage.label(m)
    return lab == (np.argmax(np.bincount(lab.ravel())[1:]) + 1) if n > 1 else m


def crop_box(ct, ts_dir: Path):
    """Index box (lo, size) in x, y, z around the crop classes; each class as its largest piece (TotalSegmentator can
    put stray 'brain' voxels far down the body)."""
    m = np.zeros(sitk.GetArrayViewFromImage(ct).shape, bool)
    for s in CROP_FROM:
        f = ts_dir / f"{s}.nii.gz"
        if not f.exists():
            raise SystemExit(f"{f} missing: run TotalSegmentator 'total' on this volume first")
        a = sitk.GetArrayFromImage(sitk.ReadImage(str(f))) > 0
        if a.shape != m.shape:
            raise SystemExit(f"{f.name}: shape {a.shape}, CT {m.shape} -- not the same volume")
        if a.any():
            m |= largest(a)
    if not m.any():
        raise SystemExit("no brain, skull or thyroid in this volume: not a head-first scan?")
    zz, yy, xx = np.nonzero(m)
    mg = [int(np.ceil(MARGIN_MM / s)) for s in ct.GetSpacing()]
    lo = [max(int(v.min()) - g, 0) for v, g in zip((xx, yy, zz), mg)]
    hi = [min(int(v.max()) + g + 1, n) for v, g, n in zip((xx, yy, zz), mg, ct.GetSize())]
    return lo, [h - l for h, l in zip(hi, lo)]


def predict(inp: Path, out: Path, weights: Path, device: str):
    env = dict(os.environ, nnUNet_results=str(weights))
    cmd = ["nnUNetv2_predict", "-i", str(inp), "-o", str(out)] + sum(([f"-{k}", v] for k, v in NNUNET.items()), []) \
        + ["-device", device]
    if device == "cpu":
        cmd += ["-npp", "1", "-nps", "1"]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, env=env, check=True)


def world_x(m, img):
    zz, yy, xx = np.nonzero(m)
    return img.TransformContinuousIndexToPhysicalPoint((float(xx.mean()), float(yy.mean()), float(zz.mean())))[0]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nifti", required=True, help="the head-first CT volume the other models saw")
    ap.add_argument("--ts", required=True, help="TotalSegmentator 'total' output folder of that volume")
    ap.add_argument("--weights", required=True, help="folder holding Dataset558_OAR258 (nnUNet_results)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    a = ap.parse_args()

    nifti, out = Path(a.nifti), Path(a.out)
    if not (Path(a.weights) / "Dataset558_OAR258").is_dir():
        raise SystemExit(f"no Dataset558_OAR258 in {a.weights}: unzip the CADS weights there")
    name = nifti.name[:-7] if nifti.name.endswith(".nii.gz") else nifti.stem
    ct = sitk.ReadImage(str(nifti), sitk.sitkInt16)
    lo, size = crop_box(ct, Path(a.ts))
    inp, pred, mdir = out / "cads_input", out / "cads_pred", out / "masks"
    for d in (inp, pred, mdir):
        d.mkdir(parents=True, exist_ok=True)
    crop = sitk.DICOMOrient(sitk.RegionOfInterest(ct, size, lo), "RAS")
    sitk.WriteImage(crop, str(inp / f"{name}_0000.nii.gz"), True)
    print(f"{name}: crop {size} voxels from {lo}", flush=True)
    predict(inp, pred, Path(a.weights), a.device)

    # nnU-Net writes the labels on the crop's geometry: back onto the full grid by world coordinates
    lab = sitk.Resample(sitk.ReadImage(str(pred / f"{name}.nii.gz")), ct, sitk.Transform(),
                        sitk.sitkNearestNeighbor, 0, sitk.sitkUInt8)
    arr = sitk.GetArrayFromImage(lab)
    masks = {s: largest(arr == v) for s, v in LABELS.items()}
    x_mid = world_x(masks["BrainStem"], ct) if masks["BrainStem"].any() else None
    vcc = float(np.prod(ct.GetSpacing())) / 1000.0
    rows = []
    for s, m in masks.items():
        side = ""
        if m.any() and x_mid is not None and s.endswith(("_L", "_R")):
            on_left = world_x(m, ct) > x_mid                 # LPS world: patient left = +x
            side = "ok" if on_left == s.endswith("_L") else "WRONG SIDE"
        if m.any():
            o = sitk.GetImageFromArray(m.astype(np.uint8))
            o.CopyInformation(ct)
            sitk.WriteImage(o, str(mdir / f"{s}.nii.gz"), True)
        rows.append(dict(structure=s, cads_label=LABELS[s], volume_cc=round(int(m.sum()) * vcc, 3),
                         left_right=side, note="" if m.any() else "empty: delineate by hand"))
        print(f"   {s:<16} {rows[-1]['volume_cc']:7.3f} cc  {side}", flush=True)
    with (out / "head_oois_cads.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if any(r["left_right"] == "WRONG SIDE" for r in rows):
        print("!! a left/right structure lies on the other side of the brainstem: check the masks", file=sys.stderr)


if __name__ == "__main__":
    main()
