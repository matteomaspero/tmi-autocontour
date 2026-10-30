"""
tmli_v2.py -- TMLI-alike structure set, version 2: v1 refined after the physician's review
(per-structure ratings and notes, returned 27-10-2026).

What changed against v1, each traced to a note (FEEDBACK below is written into the report):
  targets
    CTV_M_Skull       mandible split off ("Mandible integrated in Skull contour; needs splitting")
    CTV_M_Mandible    new ("need to have; can it be split?")
    CTV_M_Spine       3 mm closing, spinal canal kept out ("could be more coarse even")
    CTV_M_RibsShoulders  unlabelled bone between rib and vertebra added ("ribs medially against spine are cut off")
    CTV_M_Arm_L/R     humerus + forearm + hand ("CTV_M forearm + hand: need to have")
    CTV_M_Leg_L/R     femur + lower leg + foot (FF scan; the clinical CTVs run to the feet)
                      Both rebuilt from the CT: TotalSegmentator 'total' has no forearm/hand/tibia/foot, and in
                      children it calls the femur shaft 'humerus' (seen in two children). Bone = HU > 200
                      inside the patient, grown from the humeral head (at the scapula) and the femoral head (at
                      the acetabulum) across joint gaps; left/right from the body midline, not from TS's label.
    CTV_LN_Abdomen, CTV_LN_Pelvis  stand-in cropped from bone and muscle ("adjustment against surrounding
                      bone/muscle"); '_proxy' dropped, rated minor issue
    CTV_LN_Neck_proxy, CTV_LN_Thorax_proxy  cropped from bone (thorax also from lung); still drafts: "LN levels
                      are different from vessels + margin" (major issue) -- a model or manual work is needed
    CTV_Brain         new ("need to have"; = cranial contents, as Brain)
  OARs
    Eye_L/R           lens included ("may include the lenses")
    Submandibular_L/R, Bone_Mandible, Aorta, Trachea, Pancreas  new
    Gallbladder       removed ("not needed")
    every L/R pair    laterality check against the body midline (open action 13)
    head/thorax ROIs  kept only near their region (brain / lungs): no TS strays on another scan
    single organs     stray pieces dropped (largest component kept; TS 'brain' had voxels down to the thighs)
Not possible from the models that ran (listed in the report as manual / nnInteractive / other model):
BrainStem, Chiasm, Pituitary, Cochlea, Cavity_Oral, Larynx, LacrimalGland, Breast, CTV_LN_Axilla, CTV_Testes,
Rectum, Canal_Anal, Testes. Optional sources are used when their outputs exist next to the TS tasks:
TS 'craniofacial_structures' (mandible), TS 'breasts', and nnInteractive-agent objects (see nninteractive/).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage

import tmli_v1
from tmli_common import (C_, L_, LOBES_L, LOBES_R, PEDABD_VALUES, RIBS, T_, close, dilate, fill_slices,
                         keep_main)

VERSION = "v2"
TITLE = "TMLI auto v2 (after review BH 27-10-2026)"

AXIAL_BONES = (["skull", "sternum", "costal_cartilages", "scapula_left", "scapula_right", "clavicula_left",
                "clavicula_right", "hip_left", "hip_right", "sacrum", "vertebrae_S1"] + C_ + T_ + L_ + RIBS)
LONG_BONES = ["humerus_left", "humerus_right", "femur_left", "femur_right"]
MUSCLES = ["iliopsoas_left", "iliopsoas_right", "autochthon_left", "autochthon_right", "gluteus_maximus_left",
           "gluteus_maximus_right", "gluteus_medius_left", "gluteus_medius_right", "gluteus_minimus_left",
           "gluteus_minimus_right"]
BONE_HU = 200                 # cortex and the dense trabecular bone of children; marrow is filled per slice
JOINT_TOL_MM = (8.0, 15.0)    # gap bridged between bone pieces: 8 mm for any piece, 15 mm for pieces >= 0.2 cc
                              # (a 7-year-old's knee: > 8 mm of cartilage between femur and tibia)
WRIST_TOL_MM = 25.0           # arms only, after the others: a child's wrist has few ossified carpals (a 7-year-old's right
                              # hand). Legs never get it: a leg chain must not reach a hand lying on the thigh

# nnInteractive-agent object names (objects.json, case-insensitive) -> structure. Prompts: nninteractive/PROMPTS.md
NNI_NAMES = {
    "brainstem": ("BrainStem", "OAR"), "chiasm": ("Chiasm", "OAR"), "optic chiasm": ("Chiasm", "OAR"),
    "pituitary": ("Pituitary", "OAR"), "pituitary gland": ("Pituitary", "OAR"),
    "cochlea left": ("Cochlea_L", "OAR"), "cochlea right": ("Cochlea_R", "OAR"),
    "oral cavity": ("Cavity_Oral", "OAR"), "larynx": ("Larynx", "OAR"),
    "lacrimal gland left": ("LacrimalGland_L", "OAR"), "lacrimal gland right": ("LacrimalGland_R", "OAR"),
    "breast left": ("Breast_L", "OAR"), "breast right": ("Breast_R", "OAR"),
    "axillary lymph nodes left": ("CTV_LN_Axilla_L", "CTV_LN"), "axillary lymph nodes right": ("CTV_LN_Axilla_R", "CTV_LN"),
    "testes": ("Testes", "OAR"), "rectum": ("Rectum", "OAR"), "anal canal": ("Canal_Anal", "OAR"),
}

FEEDBACK = [  # structure, the physician's note, what v2 does
    ("CTV_M_Skull", "Mandible integrated in Skull contour; needs splitting if possible. Maxilla/teeth also "
     "integrated, less of an issue", "mandible split off (slice-by-slice from the chin; the top ~1 cm of the "
     "rami and some maxillary tooth buds may need a hand edit)"),
    ("CTV_M_Spine", "Could be more coarse even", "3 mm closing (spinal canal kept out)"),
    ("CTV_M_RibsShoulders", "Ribs medially against spine are cut off", "unlabelled bone between rib and vertebra added"),
    ("CTV_M_Arm_L/R", "Upper arm (good); forearm + hand: need to have", "humerus + forearm + hand from the CT (HU > 200), "
     "grown from the humeral head; couch and TBI frame excluded. A forearm/hand in a scan without its humerus (adult "
     "FF scan, arms along the body) is not assigned: edit by hand"),
    ("CTV_M_Leg_L/R", "femur good; lower legs and feet on the FF scan", "femur + lower leg + foot from the CT, grown from the femoral head"),
    ("CTV_M_Mandible / Mandible", "need to have; now integrated in Skull", "CTV_M_Mandible and Bone_Mandible (same volume)"),
    ("CTV_LN_Neck", "Good basis but neck LN levels are different from vessels+margin", "cropped from bone; still a draft (_proxy)"),
    ("CTV_LN_Thorax", "Good basis but thorax LN levels are different from vessels+margin", "cropped from bone and lung; still a draft (_proxy)"),
    ("CTV_LN_Abdomen", "Good basis; adjustment against surrounding bone/muscle", "cropped from bone and muscle"),
    ("CTV_LN_Pelvis", "Good basis; adjustment against surrounding bone/muscle", "cropped from bone and muscle"),
    ("CTV_LN_Axilla_L/R", "Need to have; difficult since no major vessels, need to train", "not made: nnInteractive prompt or manual"),
    ("CTV_LN_Pericard / Aorta boost", "Not needed!! Only for specific cases", "not made"),
    ("CTV_Brain", "need to have", "new: = Brain (cranial contents)"),
    ("Eye_L/R", "may include the lenses (now excluded)", "lens included"),
    ("Parotid_L/R", "only outer parotid delineated", "unchanged: deep lobe missing in TS, edit by hand"),
    ("Submandibular_L/R", "need to have", "new (TS head_glands_cavities)"),
    ("Esophagus", "Major issue because of tube (2 of 6)", "unchanged: check where a tube is present"),
    ("Heart", "Expands in pulmonary artery (2 adults, TS heart)", "unchanged: under 18 the PedAbd heart (rated good)"),
    ("Trachea", "Nice to have", "new (TS)"), ("Aorta", "need to have", "new (TS)"),
    ("Gallbladder", "not needed", "removed"),
    ("BrainStem, Chiasm, Pituitary, Cochlea, Cavity_Oral, Larynx, LacrimalGland, Breast", "need/nice to have",
     "no model ran for these: nnInteractive-agent (nninteractive/PROMPTS.md) or manual"),
    ("L/R pairs", "TS sometimes swaps left and right (open action 13)", "each piece checked against the body midline"),
    ("Head / thorax structures", "(seen on a child's FF scan: 'skull', 'trachea' labels on a hand and the thigh)",
     "kept only within 120 mm of the brain / 80 mm of the lungs"),
]

ORGANS_ONE_PIECE = {"Brain", "Eye_L", "Eye_R", "Lens_L", "Lens_R", "Parotid_L", "Parotid_R", "Submandibular_L",
                    "Submandibular_R", "Glnd_Thyroid", "Trachea", "Lung_L", "Lung_R", "Heart", "Aorta", "Liver",
                    "Kidney_L", "Kidney_R", "Stomach", "Pancreas", "Bladder", "CTV_LN_Spleen", "CTV_Brain"}
REGIONS = [  # (anchor TS classes, margin mm beyond the anchor's first/last slice, structures held to it)
    (["brain"], 120, {"CTV_M_Skull", "CTV_M_Mandible", "Bone_Mandible", "Brain", "CTV_Brain", "Eye_L", "Eye_R",
                      "Lens_L", "Lens_R", "OpticNerve_L", "OpticNerve_R", "Parotid_L", "Parotid_R",
                      "Submandibular_L", "Submandibular_R"}),
    (LOBES_L + LOBES_R, 80, {"Trachea", "Esophagus", "Glnd_Thyroid", "CTV_LN_Neck_proxy", "CTV_LN_Thorax_proxy",
                             "Heart", "Lung_L", "Lung_R"}),
]   # not the ribs: a 7-year-old's 12th ribs reach 96 mm below the lung base
PAIRS = [("Eye_L", "Eye_R"), ("Lens_L", "Lens_R"), ("OpticNerve_L", "OpticNerve_R"), ("Parotid_L", "Parotid_R"),
         ("Submandibular_L", "Submandibular_R"), ("Lung_L", "Lung_R"), ("Kidney_L", "Kidney_R")]


# ------------------------------------------------------------------------------------------------- mandible
def split_mandible(skull: np.ndarray, palate: np.ndarray):
    """Mandible out of TS 'skull', slice by slice from the chin upward. Below the hard palate each axial slice
    shows the mandible as its own blob (the occipital/mastoid blob lies behind it); a blob is mandible if it
    continues the mandible of the slice below, and a blob that merged with the skull base is split by the
    nearest previous-slice label. From the palate up only blobs that stay clear of it continue (the rami);
    merged ones stop there -- going further pulled maxilla and zygoma in (tested on the first prospective case, 27-10-2026)."""
    if not skull.any() or not palate.any():
        return np.zeros_like(skull)
    # stray 'skull' voxels below the chin (hyoid, a hand, noise: 1-100 voxels) would seed the split and stop it at
    # once -- 21 of 77 HF scans of the cohort had no mandible (28-10-2026). Not keep_main: a child's mandible can
    # be its own component.
    lab, n = ndimage.label(skull, ndimage.generate_binary_structure(3, 3))
    sizes = np.bincount(lab.ravel())
    skull = np.isin(lab, [i for i in range(1, n + 1) if sizes[i] >= MIN_SKULL_PIECE])
    zs = np.where(skull.any((1, 2)))[0]
    zp = np.where(palate.any((1, 2)))[0]
    zp0, zp1 = zp[0], zp[-1]
    if zs[0] >= zp0:                       # the scan does not reach below the palate
        return np.zeros_like(skull)
    pal_d = ndimage.binary_dilation(palate.any(0), iterations=4)
    mand = np.zeros_like(skull)
    mand[zs[0]] = skull[zs[0]]
    for z in range(zs[0] + 1, min(zp1 + 4, skull.shape[0])):
        lab, n = ndimage.label(skull[z])
        prev_m, prev_o = mand[z - 1], skull[z - 1] & ~mand[z - 1]
        if not prev_m.any():
            break
        near_m = None
        for i in range(1, n + 1):
            c = lab == i
            a, b = (c & prev_m).any(), (c & prev_o).any()
            if a and not b:
                if z < zp0 or not (c & pal_d).any():
                    mand[z] |= c
            elif a and b and z < zp0:
                if near_m is None:
                    _, ind = ndimage.distance_transform_edt(~skull[z - 1], return_indices=True)
                    near_m = prev_m[ind[0], ind[1]]
                mand[z] |= c & near_m
    return mand


# ------------------------------------------------------------------------------------------------- limbs
MIN_SKULL_PIECE = 500         # voxels (0.5-3 cc by grid): smaller 'skull' components are strays for the mandible split
POSTERIOR_MM = 20.0          # bone is never this far behind the most posterior model anatomy (gluteus, autochthon,
                              # occiput); the couch rails and TBI frame blocks are (an adult: 40-60 mm)


def posterior_limit(ts, shape, sp):
    """Image row behind which nothing is patient bone: the couch board and the TBI frame on it lie inside the body
    mask and their dense rails and foot blocks pass the bone threshold (an adult, 27-10-2026: a leg chain ran into a
    rail, a foot block became 'arm'). The board is too thin to find by itself; the patient lies on it, so the limit
    is the most posterior row of the model anatomy (99.9th percentile: a stray label does not move it) + 20 mm.
    Rows grow posterior in these volumes (supine); for a prone scan the limit lies outside the patient: no effect."""
    anat = ts.any_mask()
    if anat is None or not anat.any():
        return None
    rows = np.bincount(np.nonzero(anat)[1], minlength=shape[1])
    y = int(np.searchsorted(np.cumsum(rows), 0.999 * rows.sum()))
    return min(y + int(round(POSTERIOR_MM / sp[1])), shape[1])


def limb_bones(ct, ts, body, sp, sign, mid):
    """-> dict(arm_L, arm_R, leg_L, leg_R), info. See the module docstring."""
    shape = ct.shape
    vox_cc = float(np.prod(sp)) / 1000
    axial = ts.union(AXIAL_BONES, shape)
    longb = ts.union(LONG_BONES, shape)
    thr = (ct > BONE_HU) & body & ~dilate(axial, 10, sp)     # not the trunk: rib/spine fragments, calcifications
    ctop = posterior_limit(ts, shape, sp)
    if ctop is not None:
        thr[:, ctop:, :] = False                             # couch board and frame (and TS 'humerus' on a foot block)
        longb[:, ctop:, :] = False
    cand = fill_slices(thr) & ~axial | longb
    lab, n = ndimage.label(cand, ndimage.generate_binary_structure(3, 3))
    sizes = np.bincount(lab.ravel())
    has_long = set(np.unique(lab[longb])) - {0}
    keep = {i for i in range(1, n + 1) if sizes[i] * vox_cc >= 0.05} | has_long
    objs = ndimage.find_objects(lab)

    def grow(start, taken, tol, min_cc):
        cur, frontier = set(start), list(start)
        pad = [int(np.ceil(tol / s)) + 1 for s in sp]
        while frontier:
            i = frontier.pop()
            sl = objs[i - 1]
            box = tuple(slice(max(s.start - p, 0), min(s.stop + p, dim)) for s, p, dim in zip(sl, pad, shape))
            sub = lab[box]
            d = ndimage.distance_transform_edt(sub != i, sampling=sp)
            for j in np.unique(sub[(d <= tol) & (sub > 0)]):
                if j not in cur and j not in taken and j in keep and sizes[j] * vox_cc >= min_cc:
                    cur.add(j); frontier.append(j)
        return cur

    def chain(seed, taken, wrist=False):
        cur = set(seed) - taken
        for _ in range(2):
            cur = grow(cur, taken, JOINT_TOL_MM[0], 0)
            cur = grow(cur, taken, JOINT_TOL_MM[1], 0.2)
            if wrist:
                cur = grow(cur, taken, WRIST_TOL_MM, 0)
        return cur

    def seeds_at(region_names, mm):
        reg = ts.union(region_names, shape)
        if not reg.any():
            return set()
        near = dilate(reg, mm, sp)
        return (set(np.unique(lab[near & longb])) - {0}) & keep

    legs = chain(seeds_at(["hip_left", "hip_right"], 6), set())
    arms = chain(seeds_at(["scapula_left", "scapula_right"], 6), legs, wrist=True)
    # long-bone pieces nobody reached: TS 'humerus' -> arm (forearm in the FF scan), 'femur' -> leg
    hum = ts.union(["humerus_left", "humerus_right"], shape)
    rest = has_long - legs - arms
    hum_ids = {i for i in rest if (hum[objs[i - 1]] & (lab[objs[i - 1]] == i)).any()}
    arms |= chain(hum_ids, legs | arms, wrist=True)
    legs |= chain(rest - arms - hum_ids, legs | arms)
    # NOT done: forearm + hand in a scan without the humerus (FF scan, arms along the body; an adult). Tried 27-10-2026
    # (a leftover piece in its own axial body compartment, lateral of the femur): the forearm touches the thigh in
    # the body mask, and the TBI frame's side walls are compact ~60 HU compartments that pass every such test. They
    # stay unassigned and are reported (unassigned_cc): a hand edit.

    out = {k: np.zeros(shape, bool) for k in ("arm_L", "arm_R", "leg_L", "leg_R")}
    for ids, kind in ((arms, "arm"), (legs, "leg")):
        for i in ids:
            sl = objs[i - 1]
            m = lab[sl] == i
            cz, cx = np.argwhere(m).mean(0)[[0, 2]] + [sl[0].start, sl[2].start]
            side = "L" if sign * (cx - mid[int(round(cz))]) > 0 else "R"
            out[f"{kind}_{side}"][sl] |= m
    for k in out:
        out[k] = fill_slices(close(out[k], 3, sp))
    info = dict(arm_pieces=len(arms), leg_pieces=len(legs), couch_row=ctop,
                unassigned_cc=round(sum(sizes[i] for i in keep - arms - legs) * vox_cc, 1))
    return out, info


def rib_bridge(ct, ts, sp):
    """Unlabelled bone (HU > 150) within 10 mm of both a rib and a vertebra, touching the rib: the rib neck/head
    TotalSegmentator leaves out next to the spine."""
    shape = ct.shape
    ribs = ts.union(RIBS, shape)
    vert = ts.union(C_ + T_ + L_, shape)
    if not ribs.any() or not vert.any():
        return np.zeros(shape, bool)
    allb = ts.union(AXIAL_BONES + LONG_BONES, shape)
    gap = (ct > 150) & ~allb & dilate(ribs, 10, sp) & dilate(vert, 10, sp)
    lab, n = ndimage.label(gap, ndimage.generate_binary_structure(3, 3))
    touch = set(np.unique(lab[dilate(ribs, 1.5, sp) & gap])) - {0}
    return np.isin(lab, list(touch)) if touch else np.zeros(shape, bool)


def swap_sides(mL, mR, sign, mid, min_dx_vox=3):
    """Each 3D piece of a left/right pair to the side of the body midline its centroid lies on."""
    both = mL | mR
    if not both.any():
        return mL, mR, 0
    lab, n = ndimage.label(both, ndimage.generate_binary_structure(3, 3))
    newL, newR = mL.copy(), mR.copy()
    moved = 0
    for i, sl in enumerate(ndimage.find_objects(lab), 1):
        m = lab[sl] == i
        cz, cx = np.argwhere(m).mean(0)[[0, 2]] + [sl[0].start, sl[2].start]
        dx = sign * (cx - mid[int(round(cz))])
        if abs(dx) < min_dx_vox:
            continue
        if dx > 0 and (mR[sl] & m).any():
            newL[sl] |= m; newR[sl] &= ~m; moved += int((mR[sl] & m).sum())
        elif dx < 0 and (mL[sl] & m).any():
            newR[sl] |= m; newL[sl] &= ~m; moved += int((mL[sl] & m).sum())
    return newL, newR, moved


# ------------------------------------------------------------------------------------------------- build
def build(ctx):
    """ctx as tmli_v1.build, plus ct (HU), body, sign, mid, extra {task: MaskSet}, nni {name: mask}."""
    ts, ped, sp, shape, ct = ctx["ts"], ctx["ped"], ctx["sp"], ctx["shape"], ctx["ct"]
    v1 = {s["name"]: s for s in tmli_v1.build(ctx)}
    out, notes = [], ctx.setdefault("notes", [])

    def add(name, kind, mask, source, note=""):
        out.append(dict(name=name, kind=kind, mask=mask, source=source, note=note))

    # mandible: TS craniofacial_structures when it ran, else split from TS skull
    skull = ts.union(["skull"], shape)
    cf = ctx.get("extra", {}).get("craniofacial_structures")
    if cf is not None and "mandible" in cf:
        mand, msrc = cf["mandible"] & ~ts.union(C_, shape), "TS craniofacial_structures"
    else:
        mand, msrc = split_mandible(skull, ts.union(["hard_palate"], shape)), "TS skull, split at the jaw"
    limbs, info = limb_bones(ct, ts, ctx["body"], sp, ctx["sign"], ctx["mid"])
    notes.append(f"limbs: {info['arm_pieces']} arm and {info['leg_pieces']} leg bone pieces; {info['unassigned_cc']} cc "
                 f"of threshold bone outside the trunk not assigned (FF scan: forearms/hands beside the body); "
                 + (f"couch/frame excluded from image row {info['couch_row']}" if info["couch_row"] is not None
                    else "no posterior limit (no anatomy)"))
    bridge = rib_bridge(ct, ts, sp)
    vert = ts.union(C_ + T_ + L_, shape)
    bones = ts.union(AXIAL_BONES + LONG_BONES, shape) | limbs["arm_L"] | limbs["arm_R"] | limbs["leg_L"] | limbs["leg_R"]
    bones |= (ct > BONE_HU) & ctx["body"] & dilate(ts.union(AXIAL_BONES, shape), 3, sp)   # no slice fill: the
    # rib cage and the pelvis are rings, filling them would take the whole chest
    muscles = ts.union(MUSCLES, shape)
    lungs = ts.union(LOBES_L + LOBES_R, shape)

    add("CTV_M_Skull", "CTV_M", skull & ~mand, "TotalSegmentator", "mandible removed")
    if mand.any():
        add("CTV_M_Mandible", "CTV_M", mand, msrc, "check the top of the rami and the maxillary teeth")
    # closing only: a slice fill would take the spinal canal (the arch is a ring); the cord is kept out
    add("CTV_M_Spine", "CTV_M", close(vert, 3, sp) & ~ts.union(["spinal_cord"], shape), "TotalSegmentator",
        "3 mm closing (coarser)")
    add("CTV_M_RibsShoulders", "CTV_M", v1["CTV_M_RibsShoulders"]["mask"] | bridge, "TotalSegmentator + CT",
        f"rib-spine junction added ({int(bridge.sum() * np.prod(sp) / 1000)} cc)")
    add("CTV_M_Pelvis", "CTV_M", v1["CTV_M_Pelvis"]["mask"], "TotalSegmentator")
    for side in ("L", "R"):
        add(f"CTV_M_Arm_{side}", "CTV_M", limbs[f"arm_{side}"], "CT bone, grown from TS humerus",
            "humerus + forearm + hand")
    for side in ("L", "R"):
        add(f"CTV_M_Leg_{side}", "CTV_M", limbs[f"leg_{side}"], "CT bone, grown from TS femur",
            "femur + lower leg + foot")
    add("CTV_LN_Neck_proxy", "CTV_LN", v1["CTV_LN_Neck_proxy"]["mask"] & ~bones, "vessel proxy",
        "DRAFT: neck LN levels differ from vessels + margin")
    add("CTV_LN_Thorax_proxy", "CTV_LN", v1["CTV_LN_Thorax_proxy"]["mask"] & ~bones & ~lungs, "vessel proxy",
        "DRAFT: thorax LN levels differ from vessels + margin")
    add("CTV_LN_Abdomen", "CTV_LN", v1["CTV_LN_Abdomen_proxy"]["mask"] & ~bones & ~muscles, "vessel proxy",
        "aorta/IVC + 10 mm, cropped from bone and muscle")
    add("CTV_LN_Pelvis", "CTV_LN", v1["CTV_LN_Pelvis_proxy"]["mask"] & ~bones & ~muscles, "vessel proxy",
        "iliac vessels + 7 mm, cropped from bone and muscle")
    add("CTV_LN_Spleen", "CTV_LN", v1["CTV_LN_Spleen"]["mask"], "TotalSegmentator")
    add("CTV_Brain", "CTV", v1["Brain"]["mask"], "TotalSegmentator", "= Brain (cranial contents)")

    add("Brain", "OAR", v1["Brain"]["mask"], "TotalSegmentator")
    for side, s in (("L", "left"), ("R", "right")):
        add(f"Eye_{side}", "OAR", ts.union([f"eye_{s}", f"eye_lens_{s}"], shape), "TotalSegmentator", "lens included")
    for n in ("Lens_L", "Lens_R", "OpticNerve_L", "OpticNerve_R", "Parotid_L", "Parotid_R"):
        add(n, "OAR", v1[n]["mask"], v1[n]["source"], v1[n]["note"])
    for side, s in (("L", "left"), ("R", "right")):
        add(f"Submandibular_{side}", "OAR", ts.union([f"submandibular_gland_{s}"], shape), "TotalSegmentator")
    if mand.any():
        add("Bone_Mandible", "OAR", mand, msrc, "same volume as CTV_M_Mandible")
    for n in ("SpinalCord", "Glnd_Thyroid"):
        add(n, "OAR", v1[n]["mask"], v1[n]["source"], v1[n]["note"])
    add("Trachea", "OAR", ts.union(["trachea"], shape), "TotalSegmentator")
    add("Esophagus", "OAR", v1["Esophagus"]["mask"], "TotalSegmentator", "check where a feeding tube is present")
    for n in ("Lung_L", "Lung_R", "Heart"):
        add(n, "OAR", v1[n]["mask"], v1[n]["source"], v1[n]["note"])
    add("Aorta", "OAR", ts.union(["aorta"], shape), "TotalSegmentator")
    for n in ("Liver", "Kidney_L", "Kidney_R", "Stomach"):
        add(n, "OAR", v1[n]["mask"], v1[n]["source"], v1[n]["note"])
    add("Pancreas", "OAR", ts.union(["pancreas"], shape), "TotalSegmentator")
    for n in ("Duodenum", "Bowel_Small", "Bowel_Large", "Bladder"):
        add(n, "OAR", v1[n]["mask"], v1[n]["source"], v1[n]["note"])

    br = ctx.get("extra", {}).get("breasts")
    if br is not None:
        m = br.union(list(br), shape)
        xs = np.arange(shape[2])[None, None, :]
        left = ctx["sign"] * (xs - ctx["mid"][:, None, None]) > 0
        add("Breast_L", "OAR", m & left, "TS breasts")
        add("Breast_R", "OAR", m & ~left, "TS breasts")
    for oname, m in ctx.get("nni", {}).items():
        std = NNI_NAMES.get(oname.strip().lower())
        if std is None:
            notes.append(f"nnInteractive object '{oname}' has no structure name mapping: skipped")
        elif not any(s["name"] == std[0] for s in out):
            add(std[0], std[1], m, "nnInteractive-agent", "promptable model, check")

    # stray labels from another body region (a child's FF scan, feet to pelvis: TS 'skull' on a hand bone, 'trachea'
    # and 'esophagus' voxels in the thigh): a structure lives within its anchor's slices + margin, else nowhere
    for anchor, mm, names in REGIONS:
        a = ts.union(anchor, shape)
        zr = np.nonzero(a.any((1, 2)))[0]
        for s in out:
            if s["name"] not in names or not s["mask"].any():
                continue
            m = np.zeros(shape, bool)
            if len(zr):
                lo, hi = int(zr[0] - mm / sp[0]), int(zr[-1] + mm / sp[0])
                m[max(lo, 0):hi + 1] = s["mask"][max(lo, 0):hi + 1]
            if int(m.sum()) < int(s["mask"].sum()):
                notes.append(f"{s['name']}: {int(s['mask'].sum()) - int(m.sum())} voxels outside the "
                             f"{'/'.join(anchor[:1])} region removed")
                s["mask"] = m

    by = {s["name"]: s for s in out}
    for a, b in PAIRS:
        if a in by and b in by:
            by[a]["mask"], by[b]["mask"], moved = swap_sides(by[a]["mask"], by[b]["mask"], ctx["sign"], ctx["mid"])
            if moved:
                notes.append(f"laterality: {moved} voxels moved between {a} and {b}")
                by[a]["note"] = (by[a]["note"] + "; L/R corrected").strip("; ")
                by[b]["note"] = (by[b]["note"] + "; L/R corrected").strip("; ")
    for s in out:
        if s["name"] in ORGANS_ONE_PIECE:
            before = int(s["mask"].sum())
            s["mask"] = keep_main(s["mask"])
            if before - int(s["mask"].sum()) > 0.01 * max(before, 1):
                notes.append(f"{s['name']}: stray pieces removed ({before - int(s['mask'].sum())} voxels)")
    return out
