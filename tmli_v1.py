"""
tmli_v1.py -- TMLI-alike structure set, version 1: exactly the recipe a radiation oncologist reviewed on 14-10-2026
(six patients; model choice per structure of 10-10-2026). FROZEN: do not improve here -- improvements
go in tmli_v2.py, so v1 always reproduces what was reviewed (checked in-house against the reviewed sets, voxel for voxel).

One source per structure: TotalSegmentator (TS) as backbone; PedAbd for the kidneys (all ages) and the heart
under 18; lymph-node CTVs are vessel + margin stand-ins (*_proxy, a DRAFT: no model has them).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

from tmli_common import C_, L_, LOBES_L, LOBES_R, PEDABD_VALUES, RIBS, T_

VERSION = "v1"
TITLE = "TMLI auto v1 (as reviewed 14-10-2026)"
CHILD_AGE = 18.0

MODEL_CHOICE = [
    # name, kind, ts classes, ped classes, rule, reason
    ("CTV_M_Skull", "CTV_M", ["skull"], [], "TS", "only TS has bones; cortex = CTV_M border"),
    ("CTV_M_Spine", "CTV_M", C_ + T_ + L_, [], "TS", "PedAbd vertebrae no better"),
    ("CTV_M_RibsShoulders", "CTV_M", RIBS + ["sternum", "scapula_left", "scapula_right", "clavicula_left",
                                             "clavicula_right"], [], "TS", "costal cartilage left out (Hoeben 2025)"),
    ("CTV_M_Pelvis", "CTV_M", ["hip_left", "hip_right", "sacrum", "vertebrae_S1"], [], "TS", ""),
    ("CTV_M_Arm_L", "CTV_M", ["humerus_left"], [], "TS", "humerus only: no forearm/hand in TS"),
    ("CTV_M_Arm_R", "CTV_M", ["humerus_right"], [], "TS", "humerus only: no forearm/hand in TS"),
    ("CTV_M_Leg_L", "CTV_M", ["femur_left"], [], "TS", "femur only: no lower leg/foot in TS"),
    ("CTV_M_Leg_R", "CTV_M", ["femur_right"], [], "TS", "femur only: no lower leg/foot in TS"),
    ("CTV_LN_Neck_proxy", "CTV_LN", [], [], "proxy:CTV_LN_Neck", "no model: carotids + 15 mm, draft"),
    ("CTV_LN_Thorax_proxy", "CTV_LN", [], [], "proxy:CTV_LN_Thorax", "no model: mediastinal vessels + 7 mm, draft"),
    ("CTV_LN_Abdomen_proxy", "CTV_LN", [], [], "proxy:CTV_LN_Abdomen", "no model: aorta/IVC + 10 mm, draft"),
    ("CTV_LN_Pelvis_proxy", "CTV_LN", [], [], "proxy:CTV_LN_Pelvis", "no model: iliac vessels + 7 mm, draft"),
    ("CTV_LN_Spleen", "CTV_LN", ["spleen"], ["spleen"], "TS", "TS stable at all ages"),
    ("Brain", "OAR", ["brain"], [], "TS", ""),
    ("Eye_L", "OAR", ["eye_left"], [], "TS", ""),
    ("Eye_R", "OAR", ["eye_right"], [], "TS", ""),
    ("Lens_L", "OAR", ["eye_lens_left"], [], "TS", "few voxels: review"),
    ("Lens_R", "OAR", ["eye_lens_right"], [], "TS", "few voxels: review"),
    ("OpticNerve_L", "OAR", ["optic_nerve_left"], [], "TS", "thin: correct by hand"),
    ("OpticNerve_R", "OAR", ["optic_nerve_right"], [], "TS", "thin: correct by hand"),
    ("Parotid_L", "OAR", ["parotid_gland_left"], [], "TS", "TS smaller than clinical"),
    ("Parotid_R", "OAR", ["parotid_gland_right"], [], "TS", "TS smaller than clinical"),
    ("SpinalCord", "OAR", ["spinal_cord"], ["spinal_cord"], "TS", "TS covers the whole cord"),
    ("Glnd_Thyroid", "OAR", ["thyroid_gland"], [], "TS", "weak in children"),
    ("Esophagus", "OAR", ["esophagus"], [], "TS", ""),
    ("Lung_L", "OAR", LOBES_L, ["lung_left"], "TS", "TS 0.95-0.96 at all ages"),
    ("Lung_R", "OAR", LOBES_R, ["lung_right"], "TS", "TS 0.95-0.96 at all ages"),
    ("Heart", "OAR", ["heart"], ["heart"], "PED<18", "PedAbd child 0.89 vs TS 0.78; adult PedAbd 0.43"),
    ("Liver", "OAR", ["liver"], ["liver"], "TS", ""),
    ("Kidney_L", "OAR", ["kidney_left"], ["kidney_left"], "PED", "PedAbd better at all ages"),
    ("Kidney_R", "OAR", ["kidney_right"], ["kidney_right"], "PED", "PedAbd better at all ages"),
    ("Stomach", "OAR", ["stomach"], ["stomach"], "TS", "PedAbd adult 0.45"),
    ("Bowel_Large", "OAR", ["colon"], [], "TS", "PedAbd has no colon/small-bowel split"),
    ("Bowel_Small", "OAR", ["small_bowel"], [], "TS", "PedAbd has no colon/small-bowel split"),
    ("Duodenum", "OAR", ["duodenum"], [], "TS", ""),
    ("Gallbladder", "OAR", ["gallbladder"], [], "TS", "not scored yet"),
    ("Bladder", "OAR", ["urinary_bladder"], [], "TS", ""),
]
MODEL_CHOICE = [dict(zip(["name", "kind", "ts", "ped", "rule", "reason"], r)) for r in MODEL_CHOICE]

# vessel + margin stand-ins (clinical_match.RECIPES): (vessels, margin mm, cranio-caudal limit)
PROXY = {
    "CTV_LN_Neck": (["common_carotid_artery_left", "common_carotid_artery_right"], 15, None),
    "CTV_LN_Thorax": (["aorta", "superior_vena_cava", "brachiocephalic_trunk", "brachiocephalic_vein_left",
                       "brachiocephalic_vein_right", "pulmonary_vein", "trachea", "esophagus"], 7, "above_diaphragm"),
    "CTV_LN_Abdomen": (["aorta", "inferior_vena_cava"], 10, "kidneys_to_bifurcation"),
    "CTV_LN_Pelvis": (["iliac_artery_left", "iliac_artery_right", "iliac_vena_left", "iliac_vena_right"], 7, None),
}


def proxy_mask(spec, ts, spacing_zyx, shape):
    vessels, margin, zlim = spec
    seed = np.zeros(shape, bool)
    for v in vessels:
        if v in ts:
            seed |= ts[v]
    if not seed.any():
        return seed
    if zlim == "above_diaphragm":
        liver = ts.get("liver")
        if liver is not None and liver.any():
            seed[: np.where(liver.any((1, 2)))[0].max() - 5] = False       # drop the abdominal part
    elif zlim == "kidneys_to_bifurcation":
        kid = [ts[k] for k in ("kidney_left", "kidney_right") if k in ts]
        ili = [ts[k] for k in ("iliac_artery_left", "iliac_artery_right") if k in ts]
        if kid:
            ztop = max(np.where(k.any((1, 2)))[0].max() for k in kid)
            seed[ztop + 1:] = False
        if ili:
            zbif = max(np.where(i.any((1, 2)))[0].max() for i in ili)
            seed[:zbif] = False
    if not seed.any():
        return seed
    idx = np.argwhere(seed)
    pad = [int(np.ceil(margin / s)) + 1 for s in spacing_zyx]
    lo = np.maximum(idx.min(0) - pad, 0)
    hi = np.minimum(idx.max(0) + pad + 1, shape)
    box = tuple(slice(a, b) for a, b in zip(lo, hi))
    out = np.zeros(shape, bool)
    out[box] = ndimage.distance_transform_edt(~seed[box], sampling=spacing_zyx) <= margin
    return out


def auto_mask(c, ts, ped, age, spacing_zyx, shape):
    rule = c["rule"]
    if rule.startswith("proxy:"):
        return proxy_mask(PROXY[rule.split(":", 1)[1]], ts, spacing_zyx, shape), "vessel proxy"
    use_ped = rule == "PED" or (rule == "PED<18" and age is not None and age < CHILD_AGE)
    if use_ped and ped is not None:
        vals = [v for v, nm in PEDABD_VALUES.items() if nm in c["ped"]]
        return np.isin(ped, vals), "PedAbd"
    m = np.zeros(shape, bool)
    for nme in c["ts"]:
        if nme in ts:
            m |= ts[nme]
    return m, "TotalSegmentator"


def build(ctx):
    """ctx: dict(ts=MaskSet, ped=array|None, age=float|None, sp=(z,y,x) mm, shape). -> [structure dicts]"""
    out = []
    for c in MODEL_CHOICE:
        m, src = auto_mask(c, ctx["ts"], ctx["ped"], ctx["age"], ctx["sp"], ctx["shape"])
        out.append(dict(name=c["name"], kind=c["kind"], mask=m, source=src, note=c["reason"]))
    return out
