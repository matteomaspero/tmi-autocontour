# nnInteractive-agent: the structures no model gave (optional step 1b)

[nnInteractive-agent](https://github.com/MIC-DKFZ/nninteractive-agent) (DKFZ, arXiv:2610.31080) lets an
agent (Claude Code, Codex) segment a 3D image from a plain-language request. It prompts the nnInteractive
model with scribbles, boxes and points, looks at the result and corrects it. It is used here only for
structures that TotalSegmentator `total` / `head_glands_cavities` and PedAbd do not provide. The physician marked
these "need to have" or "nice to have" on 27-10-2026.

**Status: not run yet.** It is wired into `build_case.py` (`--nni-dir`) but not tried on a patient.

## Before using it on a patient

- **Images go to the LLM.** The agent's model sees rendered views of the scan. Run it on the
  **pseudonymised** NIfTI only (`<work>/nifti/<label>/...`), never on the clinical DICOM. Check with
  the department that patient images may go to the chosen LLM (Claude subscription or API, or a local
  open model behind vLLM; local is slower and less precise according to the README).
- **Model weights are CC BY-NC-SA 4.0 (non-commercial).** Confirm that your clinical use is
  covered before anything made with it is approved for treatment.
- It needs a CUDA GPU with about 10 GB. On a CPU every prompt takes long. Install it on the GPU machine, or
  run `nninteractive-server` there and point the client at it.
- Every structure it makes is a **draft**. It comes into v2 with source `nnInteractive-agent` and the note
  "promptable model, check".

## Install and run (GPU machine)

```
pip install "nninteractive-agent[local] @ git+https://github.com/MIC-DKFZ/nninteractive-agent"
nninteractive-agent setup claude          # registers the MCP server + skill for Claude Code
```

Then one request per CT volume, from the folder of the NIfTI. For the HF scan (head to thighs):

```
claude "In CT.nii.gz (a child's whole-body CT, head-first) segment each of these as its own object, using
exactly these names, and save the result to ./nni/<case>_<vol>/:
brainstem; optic chiasm; pituitary gland; cochlea left; cochlea right; oral cavity; larynx;
lacrimal gland left; lacrimal gland right; axillary lymph nodes left; axillary lymph nodes right;
breast left; breast right; testes; rectum; anal canal.
Left and right are the PATIENT's left and right. Skip an object that is not in the image and say so."
```

Leave out what does not apply to the patient (breasts in a young boy, testes in a girl). The FF scan
(feet to pelvis) needs at most: testes, rectum, anal canal.

## What build_case.py expects

`--nni-dir <dir>` with one folder per volume, named like the segmentation folders:

```
<dir>/P001_20260101_CT_2/CT_2_instances.nii.gz     one id per object, on the same grid as the NIfTI
<dir>/P001_20260101_CT_2/objects.json              names (and notes) per id
```

Object names are matched case-insensitively to structure names through `tmli_v2.NNI_NAMES`
("brainstem" → `BrainStem`, "axillary lymph nodes left" → `CTV_LN_Axilla_L`, ...). A name that is not
in the table is reported in the build notes and skipped. Add to the table rather than renaming objects.
A structure that v2 already makes from TotalSegmentator is not replaced.

## Also worth trying (revisions of existing v2 structures, physician's notes)

Not wired in: these replace a model structure, so a person decides per case.

- Parotid_L/R: "only the outer parotid delineated". Ask for the whole gland, deep lobe included.
- CTV_LN_Neck / CTV_LN_Thorax: "LN levels are different from vessels + margin". Ask for the cervical
  levels II–V / the mediastinal stations, if the doctors give a definition the agent can follow.
- Esophagus where a feeding tube is present.
