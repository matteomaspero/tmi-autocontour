#!/usr/bin/env python3
"""
segment_case.py -- step 1 of the clinical TMLI pipeline, for ONE new patient, on the GPU server.

The PedAbdSeg scripts (https://github.com/MMianyong/-PedAbdSeg-: convert_ct.py, run_totalseg.py,
run_pedabd_model.sh) run a whole data folder; this does the same steps for one case and writes the same layout, so build_case.py
(step 2) reads it unchanged:

    <work>/nifti/<pt>/<date>/CT.nii.gz, CT_2.nii.gz, ...                    one per CT series
    <work>/segmentations/totalsegmentator/<task>/<pt>_<date>_<CT>/*.nii.gz   one folder per task and volume
    <work>/segmentations/pedabd_model/<pt>_<date>_<CT>.nii.gz                PedAbd nnU-Net (Dataset201, fold 0)

Input is either the case's DICOM (one folder or several; every CT series with >= 50 slices becomes a volume)
or the GIPL files of a data export (<data>/<pt>/<date>/Gipl/*.gipl).
Use a pseudonymised export (label, e.g. P001): the patient ID never needs to reach the GPU machine.

  source /path/to/venv/bin/activate          # TotalSegmentator + nnU-Net v2 environment
  python segment_case.py --pt P001 --date 20260101 --dicom <data>/P001/20260101/Dcm \\
      --work <models folder> --gpu 0
  python segment_case.py --pt P001 --date 20260101 --gipl <data>/P001/20260101/Gipl --work ...

--tasks defaults to the two the reviewed recipe uses (total, head_glands_cavities); add craniofacial_structures
(mandible: v2 then uses it instead of splitting the skull) and breasts when wanted -- both free TS tasks.
Written 27-10-2026 from the PedAbdSeg working scripts; the in-house runs used those scripts directly, so check
one volume against them on first use.
"""
from __future__ import annotations

import argparse
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import SimpleITK as sitk


def dicom_volumes(folders):
    """Every CT series with >= 50 slices, as SimpleITK images (true geometry from the headers)."""
    out = []
    for d in folders:
        reader = sitk.ImageSeriesReader()
        for sid in reader.GetGDCMSeriesIDs(str(d)) or []:
            files = reader.GetGDCMSeriesFileNames(str(d), sid)
            if len(files) < 50:              # localizer, dose report, exam summary
                continue
            reader.SetFileNames(files)
            out.append((f"{Path(d).name}/{sid[-8:]}", reader.Execute()))
    return out


def gipl_volumes(folder):
    """As PedAbdSeg's convert_ct.py: one volume per distinct file size (duplicates are the same scan)."""
    seen, out = set(), []
    for f in sorted(glob.glob(str(Path(folder) / "*.gipl"))):
        s = os.path.getsize(f)
        if s not in seen:
            seen.add(s)
            out.append((Path(f).name[:60], sitk.ReadImage(f)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pt", required=True, help="pseudonym label, e.g. P001")
    ap.add_argument("--date", required=True, help="scan date folder, YYYYMMDD")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dicom", nargs="+")
    src.add_argument("--gipl")
    ap.add_argument("--work", required=True, help="models folder (holds PedSegAbd/nnUNet_results, the PedAbdSeg weights)")
    ap.add_argument("--tasks", nargs="+", default=["total", "head_glands_cavities"])
    ap.add_argument("--gpu", default="1")
    a = ap.parse_args()

    work = Path(a.work)
    os.environ["CUDA_VISIBLE_DEVICES"] = a.gpu
    ndir = work / "nifti" / a.pt / a.date
    ndir.mkdir(parents=True, exist_ok=True)
    vols = dicom_volumes(a.dicom) if a.dicom else gipl_volumes(a.gipl)
    if not vols:
        sys.exit("no CT volume found")
    names = []
    for i, (src_name, img) in enumerate(vols):
        name = "CT" if i == 0 else f"CT_{i + 1}"
        p = ndir / f"{name}.nii.gz"
        if not p.exists():
            sitk.WriteImage(img, str(p), True)
        names.append(name)
        print(f"{src_name} -> {p} {img.GetSize()}", flush=True)

    from totalsegmentator.python_api import totalsegmentator
    for name in names:
        case = f"{a.pt}_{a.date}_{name}"
        for task in a.tasks:
            out = work / "segmentations" / "totalsegmentator" / task / case
            if (out / ".done").exists():
                print(f"skip {task} {case}: done before", flush=True)
                continue
            out.mkdir(parents=True, exist_ok=True)
            totalsegmentator(input=str(ndir / f"{name}.nii.gz"), output=str(out), task=task)
            (out / ".done").write_text("done\n")
            print(f"{task} {case}: done", flush=True)

    # PedAbd nnU-Net, as run_pedabd_model.sh: stage <case>_0000.nii.gz, predict, move
    os.environ["nnUNet_results"] = str(work / "PedSegAbd" / "nnUNet_results")
    pdir = work / "segmentations" / "pedabd_model"
    pdir.mkdir(parents=True, exist_ok=True)
    todo = [n for n in names if not (pdir / f"{a.pt}_{a.date}_{n}.nii.gz").exists()]
    if todo:
        with tempfile.TemporaryDirectory(dir=work / "segmentations") as tmp:
            sin, sout = Path(tmp) / "in", Path(tmp) / "out"
            sin.mkdir(); sout.mkdir()
            for n in todo:
                os.symlink(ndir / f"{n}.nii.gz", sin / f"{a.pt}_{a.date}_{n}_0000.nii.gz")
            exe = shutil.which("nnUNetv2_predict") or sys.exit("nnUNetv2_predict not found: activate the venv")
            subprocess.run([exe, "-d", "201", "-i", str(sin), "-o", str(sout), "-f", "0", "-c", "3d_fullres"],
                           check=True)
            for f in sout.glob("*.nii.gz"):
                shutil.move(str(f), pdir / f.name)
                print(f"pedabd: {f.name}", flush=True)
    print(f"\nnext, on Windows: build_case.py --case {a.pt}_{a.date} --nifti-dir <...>/nifti/{a.pt}/{a.date} "
          f"--seg-root <...>/segmentations --dicom <clinical CT folders> --out <output folder>")


if __name__ == "__main__":
    main()
