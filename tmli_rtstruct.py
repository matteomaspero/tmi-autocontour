"""
tmli_rtstruct.py -- write a structure set onto the ORIGINAL clinical CT (patient identity kept), and prove it.

The RTSTRUCT references the clinical series' own Study/Series/FrameOfReference/SOP UIDs and carries the
patient and study modules copied from that CT, so it imports next to the CT already in MIM / Monaco. The CT is
never written. ApprovalStatus is UNAPPROVED: a physician reviews and approves in the clinical system.

Every ROI is read back from the written file and mapped onto the model grid: the result must equal the mask it
came from (Dice >= 0.95 for every ROI of 200 voxels or more -- about 1 cc on a 1.4 x 1.4 x 3 mm grid -- and >= 0.80 for smaller ones) or the file is not kept.
"""
from __future__ import annotations

import copy
import datetime as dt
import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pydicom
from pydicom.uid import generate_uid

from tmli_common import D8, D8_INV

PATIENT_STUDY_TAGS = [
    "SpecificCharacterSet",
    "PatientName", "PatientID", "IssuerOfPatientID", "PatientBirthDate", "PatientBirthTime", "PatientSex",
    "OtherPatientIDs", "OtherPatientIDsSequence", "OtherPatientNames", "PatientAge", "PatientSize",
    "PatientWeight", "EthnicGroup", "PatientComments",
    "StudyInstanceUID", "StudyDate", "StudyTime", "ReferringPhysicianName", "StudyID", "AccessionNumber",
    "IssuerOfAccessionNumberSequence", "StudyDescription", "InstitutionName", "InstitutionAddress",
]
INTERPRETED = {"CTV_M": "CTV", "CTV_LN": "CTV", "CTV": "CTV", "OAR": "ORGAN"}


def palette(n: int):
    import matplotlib.pyplot as plt
    cols = [plt.get_cmap(c)(i)[:3] for c, k in (("tab10", 10), ("Set1", 9), ("Dark2", 8), ("tab20", 20),
                                                 ("tab20b", 20)) for i in range(k)]
    cols = [c for c in cols if max(c) - min(c) > 0.15]
    uniq = []
    for c in cols:
        if all(sum(abs(a - b) for a, b in zip(c, u)) > 0.25 for u in uniq):
            uniq.append(c)
    return [uniq[i % len(uniq)] for i in range(n)]


def free_name(p: Path) -> Path:
    """Never overwrite: <name>.dcm, then <name>_b.dcm, _c, ..."""
    if not p.exists():
        return p
    for c in "bcdefghijklmnopqrstuvwxyz":
        q = p.with_name(f"{p.stem}_{c}{p.suffix}")
        if not q.exists():
            return q
    raise FileExistsError(p)


def write(pairs, zmap, tname, structures, out_file: Path, label: str, title: str, software: str,
          series_number: int, nshape, expand=None):
    """pairs: [(file, dataset)] of the clinical series sorted along the slice normal; zmap/tname: DICOM slice ->
    NIfTI slice; structures: [dict(name, kind, mask, ...)], masks on the NIfTI grid or, with expand(mask) ->
    full-grid mask, on a crop of it (expanded one at a time: ~45 full masks do not fit). -> (path, readback rows)"""
    expand = expand or (lambda m: m)
    from rt_utils import RTStructBuilder

    inv = D8_INV[tname]
    k0 = int(np.argmax([d.pixel_array.std() for _, d in pairs[::max(1, len(pairs) // 20)]])) * max(1, len(pairs) // 20)
    probe = pairs[k0][1].pixel_array
    if not np.array_equal(inv(D8[tname](probe)), probe):
        raise RuntimeError(f"inverse transform {tname} does not invert")
    tmp = Path(tempfile.mkdtemp(prefix="tmli_ct_"))
    try:
        for f, _ in pairs:                                  # one series only (rt_utils walks folders recursively)
            shutil.copy2(f, tmp / os.path.basename(f))
        b = RTStructBuilder.create_new(dicom_series_path=str(tmp))
        k_of_sop = {str(d.SOPInstanceUID): k for k, (_, d) in enumerate(pairs)}
        order = [k_of_sop[str(d.SOPInstanceUID)] for d in b.series_data]
        empty = np.zeros(pairs[0][1].pixel_array.shape, bool)
        cols = palette(len(structures))
        added = []
        for s, col in zip(structures, cols):
            if not s["mask"].any():
                continue
            m = expand(s["mask"])
            if not m.any():
                continue
            arr = np.stack([inv(m[zmap[k]]) if zmap[k] >= 0 else empty for k in order], axis=-1)
            if not arr.any():
                continue
            b.add_roi(mask=arr, name=s["name"], color=[int(255 * c) for c in col], approximate_contours=True)
            added.append(s)
        ds = b.ds
        ct = pairs[0][1]
        for kw in PATIENT_STUDY_TAGS:                       # identity and study exactly as on the clinical CT
            if kw in ct:
                ds.add(copy.deepcopy(ct[kw]))
            elif kw in ds:
                del ds[kw]
        now = dt.datetime.now()
        ds.SeriesInstanceUID = generate_uid()
        ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = generate_uid()
        ds.SeriesNumber = series_number
        ds.SeriesDescription = title[:64]
        ds.SeriesDate, ds.SeriesTime = now.strftime("%Y%m%d"), now.strftime("%H%M%S")
        ds.StructureSetLabel = label[:16]
        ds.StructureSetName = title[:64]
        ds.StructureSetDescription = "Automatic contours, UNAPPROVED: review and approve before clinical use"
        ds.Manufacturer = "UMC Utrecht Radiotherapy"
        ds.ManufacturerModelName = "TMLI auto-contour pipeline"
        ds.SoftwareVersions = software[:64]
        ds.ApprovalStatus = "UNAPPROVED"
        kind = {s["name"]: s["kind"] for s in added}
        num_name = {r.ROINumber: r.ROIName for r in ds.StructureSetROISequence}
        for r in ds.StructureSetROISequence:
            r.ROIGenerationAlgorithm = "AUTOMATIC"
        for o in ds.RTROIObservationsSequence:
            o.RTROIInterpretedType = INTERPRETED.get(kind.get(num_name.get(o.ReferencedROINumber), "OAR"), "ORGAN")
        out_file = free_name(Path(out_file))
        out_file.parent.mkdir(parents=True, exist_ok=True)
        part = out_file.with_suffix(".partial")
        ds.save_as(str(part), write_like_original=False)

        rows = verify(tmp, part, pairs, zmap, tname, added, nshape, expand)
        bad = [r for r in rows if not r["ok"]]
        if bad:
            raise RuntimeError(f"{out_file.name}: read-back failed for {[(r['roi'], r['dice']) for r in bad]}; "
                               f"left as {part.name}")
        part.replace(out_file)
        return out_file, rows
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify(ct_dir: Path, rs_file: Path, pairs, zmap, tname, added, nshape, expand=lambda m: m):
    """References, identity and every ROI's shape, read back from the file as a TPS would."""
    from rt_utils import RTStructBuilder
    rs = pydicom.dcmread(str(rs_file))
    ct = pairs[0][1]
    for kw in ("PatientID", "PatientName", "PatientBirthDate", "PatientSex", "StudyInstanceUID"):
        if str(rs.get(kw, "")) != str(ct.get(kw, "")):
            raise RuntimeError(f"{kw} differs from the CT")
    ref = rs.ReferencedFrameOfReferenceSequence[0]
    if ref.FrameOfReferenceUID != ct.FrameOfReferenceUID:
        raise RuntimeError("FrameOfReferenceUID differs from the CT")
    sers = ref.RTReferencedStudySequence[0].RTReferencedSeriesSequence[0]
    if sers.SeriesInstanceUID != ct.SeriesInstanceUID:
        raise RuntimeError("referenced series is not the CT series")
    sops = {str(d.SOPInstanceUID) for _, d in pairs}
    refd = {str(c.ReferencedSOPInstanceUID) for c in sers.ContourImageSequence}
    if refd != sops:
        raise RuntimeError(f"contour image references != CT slices ({len(refd)} vs {len(sops)})")
    for rc in rs.ROIContourSequence:
        for c in getattr(rc, "ContourSequence", []):
            if str(c.ContourImageSequence[0].ReferencedSOPInstanceUID) not in sops:
                raise RuntimeError("a contour references an image outside the CT series")
    rt = RTStructBuilder.create_from(dicom_series_path=str(ct_dir), rt_struct_path=str(rs_file))
    k_of_sop = {str(d.SOPInstanceUID): k for k, (_, d) in enumerate(pairs)}
    order = [k_of_sop[str(d.SOPInstanceUID)] for d in rt.series_data]
    t = D8[tname]
    vox = None
    rows = []
    for s in added:
        rm = rt.get_roi_mask_by_name(s["name"])
        back = np.zeros(nshape, bool)
        for j, k in enumerate(order):
            if zmap[k] >= 0:
                back[zmap[k]] = t(rm[:, :, j])
        src = expand(s["mask"])
        inter, tot = int((back & src).sum()), int(back.sum()) + int(src.sum())
        dice = 2 * inter / tot if tot else 1.0
        n = int(src.sum())
        rows.append(dict(roi=s["name"], voxels=n, dice=round(dice, 4), ok=dice >= (0.95 if n >= 200 else 0.80)))
    return rows
