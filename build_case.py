#!/usr/bin/env python3
"""
build_case.py -- TMLI-alike structure sets for one patient, as RTSTRUCTs on the clinical CT.

Step 2 of the clinical pipeline (step 1, the models, runs on a GPU machine: segmentation/segment_case.py).
Per CT volume the models segmented (HF and FF scan):
  1. find the clinical DICOM series it was made from, pixel for pixel (never a header match);
  2. build the structure set of each version asked for -- v1 = as reviewed by a radiation oncologist on 14-10-2026
     (tmli_v1.py, frozen), v2 = refined after that review (tmli_v2.py);
  3. write per version, under --out:
        TMLI_auto_v1_asReviewed_<date>/      TMLI_auto_v2_afterReview_<date>/
            RTSTRUCT/RS_TMLI_auto_<v>_<HF|FF>_<date>.dcm   on the clinical CT, patient identity kept, UNAPPROVED
            overview/overview_<v>_<HF|FF>_<date>.png       check picture
            masks/<HF|FF>/<structure>.nii.gz               on the model grid (traceability)
            report/structures_<date>.csv, report/README_<date>.md   volumes, read-back Dice, notes, sources
     A file that exists is never overwritten (_b, _c, ...).

  python build_case.py --case P001_20260101 \\
      --nifti-dir <work>/nifti/P001/20260101 \\
      --seg-root  <work>/segmentations \\
      --dicom <folder with the FF CT series> <folder with the HF CT series> --out <output folder>

--case is the prefix of the segmentation folders (<case>_<CT volume>, as segmentation/segment_case.py names them).
Optional model outputs are used when present: <seg-root>/totalsegmentator/craniofacial_structures|breasts/
<case>_<vol>/ and --nni-dir <dir>/<case>_<vol>/ (nnInteractive-agent: *_instances.nii.gz + objects.json).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tmli_render                                                     # noqa: E402
import tmli_rtstruct                                                   # noqa: E402
import tmli_v1                                                         # noqa: E402
import tmli_v2                                                         # noqa: E402
from tmli_common import (MIDLINE, MaskSet, body_mask, find_series_for, load_masks, midline_x,  # noqa: E402
                         read_ped, side_sign)

VERSIONS = {"v1": (tmli_v1, "asReviewed", 901), "v2": (tmli_v2, "afterReview", 902)}
OPTIONAL_TASKS = ["craniofacial_structures", "breasts"]
PIPELINE_DATE = "2026-10-27"


def age_of(ds, override):
    if override is not None:
        return float(override)
    a = str(ds.get("PatientAge", "") or "")
    if len(a) == 4 and a[:3].isdigit():
        return float(a[:3]) * {"Y": 1, "M": 1 / 12, "W": 1 / 52, "D": 1 / 365}[a[3]]
    bd, sd = str(ds.get("PatientBirthDate", "")), str(ds.get("StudyDate", ""))
    if len(bd) == 8 and len(sd) == 8:
        return (dt.date(int(sd[:4]), int(sd[4:6]), int(sd[6:])) - dt.date(int(bd[:4]), int(bd[4:6]), int(bd[6:]))).days / 365.25
    raise ValueError("no age in the CT header: give --age (it selects the PedAbd heart under 18)")


def load_nni(d: Path, shape):
    """nnInteractive-agent output: <image>_instances.nii.gz (one id per object) + objects.json (names)."""
    if d is None or not d.is_dir():
        return {}
    inst = sorted(d.glob("*_instances.nii.gz"))
    meta = d / "objects.json"
    if not inst or not meta.exists():
        return {}
    lab = sitk.GetArrayFromImage(sitk.ReadImage(str(inst[0])))
    if lab.shape != tuple(shape):
        raise ValueError(f"{inst[0].name}: shape {lab.shape}, CT {shape} -- run nnInteractive on the same NIfTI")
    objs = json.loads(meta.read_text(encoding="utf-8"))
    objs = objs.get("objects", objs) if isinstance(objs, dict) else objs
    out = {}
    for o in objs:
        i = o.get("id", o.get("instance_id"))
        name = o.get("name", o.get("label"))
        if i is not None and name:
            out[str(name)] = lab == int(i)
    return out


def crop_box(body, margin=6):
    idx = [np.nonzero(body.any(a))[0] for a in ((1, 2), (0, 2), (0, 1))]
    return tuple(slice(max(int(i[0]) - margin, 0), min(int(i[-1]) + margin + 1, n)) for i, n in zip(idx, body.shape))


class Full:
    """A structure mask on the cropped grid, presented on the full grid one at a time (memory)."""

    def __init__(self, shape, box):
        self.shape, self.box = shape, box

    def __call__(self, m):
        f = np.zeros(self.shape, bool)
        f[self.box] = m
        return f


def run(a, volumes=None):
    """volumes: [(NIfTI path, segmentation case, PedAbd case)]; default every NIfTI in --nifti-dir, both cases
    <--case>_<volume> (a cohort driver can pass its own, e.g. for volumes that carry an nnU-Net _0000 suffix)."""
    date = a.date
    if volumes is None:
        volumes = [(p, f"{a.case}_{p.name[:-7]}", f"{a.case}_{p.name[:-7]}")
                   for p in sorted(Path(a.nifti_dir).glob("*.nii.gz"))]
    if not volumes:
        raise SystemExit(f"no NIfTI in {a.nifti_dir}")
    seg = Path(a.seg_root)
    ped_dir = Path(a.ped_dir) if getattr(a, "ped_dir", None) else seg / "pedabd_model"
    rows = {v: [] for v in a.versions}
    notes_all = {v: [] for v in a.versions}
    inputs = []
    for npath, case, ped_case in volumes:
        print(f"== {case}", flush=True)
        nii = sitk.ReadImage(str(npath))
        nvol = sitk.GetArrayFromImage(nii).astype(np.float32)
        shape = nvol.shape
        pairs, zmap, tname = find_series_for(nvol, a.dicom)
        ds0 = pairs[0][1]
        pos = str(ds0.get("PatientPosition", ""))
        scan = "HF" if pos.startswith("HF") else "FF" if pos.startswith("FF") else (pos or npath.name[:-7])
        if any(i["scan"] == scan for i in inputs):       # two HF scans in one session (seen in the cohort)
            scan = f"{scan}_{npath.name[:-7]}"
        age = age_of(ds0, a.age)
        print(f"   clinical series: {len(pairs)} slices, {pos}, '{ds0.get('SeriesDescription', '')}', transform "
              f"{tname}, {int((zmap < 0).sum())} DICOM slices without a NIfTI twin; age {age:.1f} y", flush=True)
        ts_dir = lambda task: seg / "totalsegmentator" / task / case
        ts = load_masks(ts_dir("total"), shape).merged(load_masks(ts_dir("head_glands_cavities"), shape))
        if not len(list(ts)):
            raise SystemExit(f"{case}: no TotalSegmentator output under {seg}")
        ped = read_ped(ped_dir / f"{ped_case}.nii.gz", shape)
        extra = {t: load_masks(ts_dir(t), shape) for t in OPTIONAL_TASKS if ts_dir(t).is_dir()}
        nni = load_nni(Path(a.nni_dir) / case if a.nni_dir else None, shape)
        anat = ts.any_mask()
        body = body_mask(nii, anat)
        box = crop_box(body)
        full = Full(shape, box)
        crop = lambda arr: None if arr is None else arr[box]
        ctx = dict(ts=MaskSet([(lab[box], idx) for lab, idx in ts.layers]), ped=crop(ped), age=age,
                   sp=tuple(nii.GetSpacing()[::-1]), shape=nvol[box].shape, ct=nvol[box], body=body[box],
                   extra={t: MaskSet([(lab[box], idx) for lab, idx in m.layers]) for t, m in extra.items()},
                   nni={k: m[box] for k, m in nni.items()})
        del ts, ped, extra, nni
        ctx["sign"] = side_sign(ctx["ts"], ctx["shape"])
        ctx["mid"] = midline_x(ctx["body"], ctx["ts"].union(MIDLINE, ctx["shape"]))
        inputs.append(dict(scan=scan, nifti=str(npath), dicom_folder=str(Path(pairs[0][0]).parent),
                           series=str(ds0.SeriesInstanceUID), slices=len(pairs), transform=tname,
                           optional=", ".join(sorted(k for k in ("craniofacial_structures", "breasts") if k in ctx["extra"]))
                           + (", nnInteractive" if ctx["nni"] else "")))
        anat_c = anat[box]
        for v in a.versions:
            mod, suffix, ser_no = VERSIONS[v]
            ctx["notes"] = []
            structs = mod.build(ctx)
            vdir = Path(a.out) / f"TMLI_auto_{v}_{suffix}_{date}"
            if a.masks:
                mdir = vdir / "masks" / scan
                mdir.mkdir(parents=True, exist_ok=True)
                for s in structs:
                    if s["mask"].any():
                        im = sitk.GetImageFromArray(full(s["mask"]).astype(np.uint8)); im.CopyInformation(nii)
                        sitk.WriteImage(im, str(mdir / f"{s['name']}.nii.gz"), True)
            rs, rb = tmli_rtstruct.write(
                pairs, zmap, tname, structs,
                vdir / "RTSTRUCT" / f"RS_TMLI_auto_{v}_{scan}_{date}.dcm", label=f"TMLI_AUTO_{v.upper()}",
                title=f"{mod.TITLE} {scan}", series_number=ser_no, nshape=shape,
                software=f"tmli_{v} {PIPELINE_DATE}; TotalSegmentator + PedAbd Dataset201", expand=full)
            print(f"   {v}: {rs.name}, {len(rb)} ROIs, read-back Dice min {min(r['dice'] for r in rb):.3f}", flush=True)
            tmli_render.render(ctx["ct"], ctx["sp"], ctx["body"], anat_c, structs,
                               f"{mod.TITLE}, {scan} scan ({len(rb)} structures)",
                               vdir / "overview" / f"overview_{v}_{scan}_{date}.png")
            dice = {r["roi"]: r["dice"] for r in rb}
            vcc = float(np.prod(ctx["sp"])) / 1000
            for s in structs:
                z = np.nonzero(s["mask"].any((1, 2)))[0]
                rows[v].append(dict(scan=scan, structure=s["name"], kind=s["kind"], source=s["source"],
                                    volume_cc=round(int(s["mask"].sum()) * vcc, 1),
                                    slices=f"{z[0] + box[0].start}-{z[-1] + box[0].start}" if len(z) else "",
                                    in_rtstruct="yes" if s["name"] in dice else "no (empty)",
                                    readback_dice=dice.get(s["name"], ""), note=s.get("note", "")))
            notes_all[v] += [f"{scan}: {n}" for n in ctx["notes"]]
            del structs
        del ctx
    for v in a.versions:
        write_report(Path(a.out) / f"TMLI_auto_{v}_{VERSIONS[v][1]}_{date}", v, rows[v], notes_all[v], inputs, date)
    return rows, notes_all, inputs


def write_report(vdir: Path, v, rows, notes, inputs, date):
    rdir = vdir / "report"
    rdir.mkdir(parents=True, exist_ok=True)
    p = tmli_rtstruct.free_name(rdir / f"structures_{date}.csv")
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    mod = VERSIONS[v][0]
    L = [f"# {mod.TITLE}", "",
         f"Generated {dt.datetime.now():%d-%m-%Y %H:%M} by build_case.py ({PIPELINE_DATE}).",
         "Automatic contours: UNAPPROVED. A physician reviews, edits and approves them in the clinical system.", "",
         "## Inputs", ""]
    for i in inputs:
        L.append(f"- {i['scan']}: {i['slices']} CT slices in `{i['dicom_folder']}` (series ...{i['series'][-12:]}), "
                 f"matched pixel for pixel to `{Path(i['nifti']).name}` ({i['transform']})"
                 + (f"; optional sources: {i['optional']}" if i["optional"] else ""))
    L += ["", "## Checks", "",
          "- RTSTRUCT: patient, study and frame of reference copied from / equal to the clinical CT; every contour "
          "references a slice of that series.",
          f"- Every ROI read back from the written file onto the model grid: lowest Dice "
          f"{min(float(r['readback_dice']) for r in rows if r['readback_dice'] != ''):.3f} "
          "(must be >= 0.95, or >= 0.80 below ~1 cc).", "",
          "## Structures", "", "See `structures_*.csv` (volume, slices, source, read-back Dice, note per scan).", ""]
    if notes:
        L += ["## Notes from the build", ""] + [f"- {n}" for n in notes] + [""]
    if v == "v2":
        L += ["## Physician review (27-10-2026) and what v2 does", "", "| Structure | Physician's note | v2 |",
              "| --- | --- | --- |"]
        L += [f"| {a} | {b} | {c} |" for a, b, c in tmli_v2.FEEDBACK]
    else:
        L += ["## v1", "", "The recipe the physician reviewed on 14-10-2026, unchanged. Known on this "
              "recipe: lymph-node CTVs are vessel + margin drafts; arm/leg CTVs are humerus/femur only, and in "
              "children TotalSegmentator calls part of the femur shaft 'humerus' -- see v2."]
    p2 = tmli_rtstruct.free_name(rdir / f"README_{date}.md")
    p2.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"report: {p}, {p2}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--case", required=True, help="segmentation prefix, e.g. P001_20260101")
    ap.add_argument("--nifti-dir", required=True)
    ap.add_argument("--seg-root", required=True)
    ap.add_argument("--ped-dir", help="PedAbd predictions; default <seg-root>/pedabd_model")
    ap.add_argument("--dicom", nargs="+", required=True, help="folders holding the clinical CT series (not recursive)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--versions", nargs="+", default=["v1", "v2"], choices=list(VERSIONS))
    ap.add_argument("--date", default=f"{dt.date.today():%Y%m%d}")
    ap.add_argument("--age", type=float, help="years; default from the CT header")
    ap.add_argument("--nni-dir", help="nnInteractive-agent outputs, one folder per <case>_<vol>")
    ap.add_argument("--no-masks", dest="masks", action="store_false", help="skip the NIfTI masks")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
