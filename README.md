# tmi-autocontour

Total marrow irradiation (TMI) structure sets from open-source segmentation models, written as a DICOM
RTSTRUCT on the patient's own planning CT.

Open whole-body models such as [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) segment more than
a hundred anatomical classes, but none of them gives the structures a TMI plan needs: bone-marrow CTVs per
region, a CNS target and the organs of interest. This code turns the outputs of two open-source models into that
set with a rule-based **recipe**. No model is retrained: the recipe works on the models' output.

- **v1**: the recipe a radiation oncologist reviewed per structure on six patients (14-10-2026). Frozen.
- **v2**: v1 refined after that review (27-10-2026). Every change and the review note it answers is in
  `tmli_v2.FEEDBACK` and is written into each run's report.

> **Research code, not a medical device.** The output is **unapproved automatic contouring**. A physician must
> review, edit and approve every structure in the clinical system before any use in treatment planning.

## Pipeline

| step | where | script | what |
| --- | --- | --- | --- |
| 1 | GPU machine | `segmentation/segment_case.py` | one NIfTI per CT series; TotalSegmentator tasks `total` + `head_glands_cavities` (optionally `craniofacial_structures`, `breasts`); the pediatric abdominal nnU-Net of [PedAbdSeg](https://github.com/MMianyong/-PedAbdSeg-) (Dataset201) |
| 1b | GPU machine, optional | `nninteractive/PROMPTS.md` | [nnInteractive-agent](https://github.com/MIC-DKFZ/nninteractive-agent) for structures no model gives. **Not validated** |
| 2 | any machine | `build_case.py` | match each NIfTI to its clinical DICOM series pixel for pixel, build v1 and/or v2, write and verify the RTSTRUCTs, overview pictures and a report |
| 3 | clinic | | import the RTSTRUCT next to the CT, then review, edit and approve |

TBI/TMI CTs are usually acquired in two parts, **head-first** (head to thighs) and **feet-first** (feet to
pelvis). Each part is handled as its own volume, and each gets its own RTSTRUCT.

## The structure set (v2)

- **Bone-marrow CTVs:** skull, mandible, spine, ribs/shoulders, pelvis, left and right arm (humerus + forearm +
  hand) and left and right leg (femur + lower leg + foot). TotalSegmentator `total` has no forearm, hand, tibia or
  foot, and in children it labels part of the femoral shaft as humerus. So the limbs are rebuilt from the CT
  (bone > 200 HU inside the patient), grown from the humeral and femoral heads across the joints, and assigned
  left or right by the patient midline.
- **CNS target:** CTV brain (the cranial contents).
- **Organs of interest:** brain, eyes (lens included), lenses, optic nerves, parotid and submandibular glands,
  spinal cord, thyroid, trachea, esophagus, lungs, heart, aorta, liver, kidneys, stomach, pancreas, duodenum,
  small and large bowel, bladder. Under 18 the heart comes from the pediatric nnU-Net; the kidneys come from it
  at every age.
- **Lymph-node regions (for TMLI):** neck, thorax, abdomen and pelvis as vessel-plus-margin stand-ins, cropped
  from bone and muscle, plus the spleen. They are **drafts**: the neck and thorax versions keep the `_proxy`
  name, because the nodal levels differ from vessel + margin.

Not produced, so they still need manual contouring: brainstem, chiasm, pituitary, cochlea, oral cavity, larynx,
lacrimal glands, breasts, axillary nodes, testes, rectum, anal canal, and the deep lobe of the parotid.

## Install

Step 2 (`build_case.py`) needs Python 3.10 or later:

```bash
pip install -r requirements.txt
```

Step 1 needs a CUDA environment with TotalSegmentator and nnU-Net v2 (see `requirements-segmentation.txt`), plus
the PedAbdSeg weights from its repository.

## Run

```bash
# step 1, GPU machine: pseudonymised CT in, model outputs under <work>
python segmentation/segment_case.py --pt P001 --date 20260101 --dicom <data>/P001/20260101/Dcm \
    --work <models folder> --gpu 0

# step 2: the clinical CT series (one folder per series) and the model outputs
python build_case.py --case P001_20260101 \
    --nifti-dir <work>/nifti/P001/20260101 --seg-root <work>/segmentations \
    --dicom <folder with the FF series> <folder with the HF series> --out <output folder>
```

Options: `--versions v2` (default both), `--age` (default: from the CT header; it selects the heart model), and
`--nni-dir` (nnInteractive-agent outputs). `--no-masks` skips the NIfTI masks.

Output, per version:

```text
TMLI_auto_v2_afterReview_<date>/
    RTSTRUCT/RS_TMLI_auto_v2_<HF|FF>_<date>.dcm    on the clinical CT, label TMLI_AUTO_V2, UNAPPROVED
    overview/overview_v2_<HF|FF>_<date>.png        check picture: coronal targets | OOIs, sagittal, axials
    masks/<HF|FF>/<structure>.nii.gz               the same structures on the model grid
    report/README_<date>.md, structures_<date>.csv inputs, checks, per structure: volume, slices, source, note
```

An existing file is never overwritten: a new name gets `_b`, `_c`, and so on.

## What every run checks

- **The CT is the one the models saw.** Every NIfTI slice must equal its DICOM slice pixel for pixel, under one
  in-plane transform. A header match alone is never trusted.
- **The RTSTRUCT belongs to the patient and the CT.** The patient and study tags are copied from the CT and
  compared again after writing. The frame of reference is the CT's, and every contour references a slice of that
  series.
- **Every ROI reads back.** Each ROI is read from the written file the way a TPS reads it (rt-utils), mapped onto
  the model grid, and compared with its source mask. It must reach Dice >= 0.95, or >= 0.80 below about 1 cc;
  otherwise the file stays `.partial`.

## Known limitations

- **Couch board and TBI frame:** their rails and foot blocks pass the bone threshold. Nothing more than 20 mm
  behind the most posterior model anatomy is accepted as bone, and the body mask keeps only the pieces that hold
  model anatomy.
- **Forearm or hand in a scan without its humerus:** for example a feet-first adult scan with the arms along the
  body. This bone is reported as unassigned; edit by hand.
- **Mandible:** it is split from the TotalSegmentator skull slice by slice, from the chin up. The top ~1 cm of
  the rami stays with the skull, and some maxillary tooth buds go with the mandible.
- **Infants:** a knee can be wider than the 15 mm joint tolerance, and then the lower leg stops at the knee.
- **Stray model labels** can appear on another body region (for example 'skull' on a hand). Head and thorax
  structures are kept only near the brain and the lungs, single organs keep their largest piece, and small pieces
  are ignored when the midline is computed.
- **The pediatric nnU-Net's label values** follow `regions_class_order`, not the `dataset.json` numbers
  (`tmli_common.PEDABD_VALUES`).

## Models and licenses

This repository contains no model weights. Check each license before clinical use:

- TotalSegmentator: code under Apache-2.0; some tasks have their own license terms (see its README).
- PedAbdSeg: see [its repository](https://github.com/MMianyong/-PedAbdSeg-).
- nnInteractive: the weights are CC BY-NC-SA 4.0 (non-commercial).

## How the recipe was developed

v1 was tuned against the clinical contours of two total marrow and lymphoid irradiation (TMLI) patients. A
radiation oncologist then rated every structure on six patients. The v2 changes were written from those notes
with an agentic AI framework (Claude Code, Anthropic), supervised by a physicist. They were regression-tested,
so v1 is unchanged voxel for voxel, and v2 was run on a TBI cohort to expose failure modes. These validation
scripts run on the clinical data in-house and are not part of this repository.

## Publication

This work has been submitted as an abstract to ESTRO 2027 (Milan, 21-25 May 2027):

**Open-source whole-body models for total marrow irradiation contouring: physician-in-the-loop refinement with an agentic AI framework**

Matteo Maspero¹, Mianyong Ding²˒¹, Enrica Seravalli¹, Geert O.R. Janssens¹˒², Bianca A.W. Hoeben¹

¹ Radiotherapy, UMC Utrecht, Utrecht, the Netherlands<br>
² Princess Máxima Center for Pediatric Oncology, Utrecht, the Netherlands

If you use this code, please cite the abstract (reference to follow once it is published) and the models it
builds on (references 1 and 2).

## References

1. Wasserthal J, et al. TotalSegmentator: robust segmentation of 104 anatomic structures in CT images.
   Radiol Artif Intell 2023;5:e230024.
2. Ding M, et al. Deep learning-based auto-contouring of organs/structures-at-risk for pediatric upper abdominal
   radiotherapy. Radiother Oncol 2025;208:110914.

## License

Apache License 2.0, see `LICENSE`.
