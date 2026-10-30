"""
tmli_common.py -- shared pieces of the clinical TMLI pipeline: model outputs, the NIfTI <-> DICOM slice map,
body mask, morphology helpers.

Frozen copies of the in-house research code (October 2026), so a change there for a study can never change
what goes to a patient. Behaviour is unchanged; only paths became arguments.
"""
from __future__ import annotations

import glob
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk
from scipy import ndimage

POOL = ThreadPoolExecutor(16)          # SMB latency, not CPU, is the limit: read files concurrently

# value in the PedAbd label map -> class name. NOT the dataset.json numbers: nnU-Net region training writes
# regions_class_order[i] (Dataset201_umcunew, checked 27-10-2026 against the model on the GPU server)
PEDABD_VALUES = {
    17: "spleen", 16: "kidney_right", 15: "kidney_left", 14: "heart", 13: "liver",
    12: "lung_right", 11: "lung_left", 10: "pancreas", 9: "bowel_bag", 19: "vertebrae",
    18: "spinal_cord", 8: "aorta", 7: "inferior_vena_cava", 6: "autochthon_left",
    5: "autochthon_right", 4: "iliopsoas_left", 3: "iliopsoas_right", 2: "stomach",
    1: "intestines",
}

C_ = [f"vertebrae_C{i}" for i in range(1, 8)]
T_ = [f"vertebrae_T{i}" for i in range(1, 13)]
L_ = [f"vertebrae_L{i}" for i in range(1, 6)]
RIBS = [f"rib_{s}_{i}" for s in ("left", "right") for i in range(1, 13)]
LOBES_L = ["lung_upper_lobe_left", "lung_lower_lobe_left"]
LOBES_R = ["lung_upper_lobe_right", "lung_middle_lobe_right", "lung_lower_lobe_right"]


# ------------------------------------------------------------------------------------------- model outputs
class MaskSet:
    """Read-only {name: bool mask} view over int16 label volumes (one per TotalSegmentator task).

    A task writes non-overlapping masks, so one label volume holds them all: ~190 MB instead of ~100
    full-size bool arrays (~9 GB per case). Masks are derived on access (lab == value)."""

    def __init__(self, layers=None):
        self.layers = layers or []

    def _find(self, n):
        for lab, idx in self.layers:
            if n in idx:
                return lab, idx[n]
        return None, None

    def __getitem__(self, n):
        lab, v = self._find(n)
        if lab is None:
            raise KeyError(n)
        return lab == v

    def get(self, n, default=None):
        return self[n] if n in self else default

    def __contains__(self, n):
        return self._find(n)[0] is not None

    def __iter__(self):
        for _, idx in self.layers:
            yield from idx

    def any_mask(self):
        out = None
        for lab, _ in self.layers:
            out = lab > 0 if out is None else (out | (lab > 0))
        return out

    def union(self, names, shape):
        m = np.zeros(shape, bool)
        for n in names:
            if n in self:
                m |= self[n]
        return m

    def merged(self, other: "MaskSet") -> "MaskSet":
        return MaskSet(self.layers + other.layers)


def _read(f):
    return sitk.GetArrayFromImage(sitk.ReadImage(str(f)))


def load_masks(d: Path, shape) -> MaskSet:
    """All masks of one task folder as one label volume. Only a 0-byte file is skipped (a truncated copy);
    any other read failure is retried once and then raised: a failed read must not silently drop a structure."""
    lab = np.zeros(shape, np.int16)
    idx = {}
    if d is None or not Path(d).is_dir():
        return MaskSet([(lab, idx)])
    files = sorted(Path(d).glob("*.nii.gz"))
    empty = [f for f in files if f.stat().st_size == 0]
    for f in empty:
        print(f"  0-byte file, skipped: {Path(d).name}/{f.name}", flush=True)
    files = [f for f in files if f not in empty]
    with ThreadPoolExecutor(8) as pool:
        for i in range(0, len(files), 8):
            chunk = files[i:i + 8]
            futs = [pool.submit(_read, f) for f in chunk]
            for f, fut in zip(chunk, futs):
                try:
                    a = fut.result()
                except RuntimeError:
                    a = _read(f)
                if a.shape != tuple(shape):
                    raise ValueError(f"{f} has shape {a.shape}, CT has {shape}")
                m = a > 0
                if m.any():
                    idx[f.name[:-7]] = len(idx) + 1
                    lab[m] = idx[f.name[:-7]]
    return MaskSet([(lab, idx)])


def read_ped(path: Path, shape):
    if path is None or not Path(path).exists():
        return None
    a = _read(path).astype(np.uint8)
    if a.shape != tuple(shape):
        raise ValueError(f"{path} has shape {a.shape}, CT has {shape}")
    return a


# ------------------------------------------------------------------------------------------- DICOM <-> NIfTI
D8 = {      # DICOM slice -> NIfTI slice
    "id": lambda a: a, "flipud": lambda a: a[::-1], "fliplr": lambda a: a[:, ::-1],
    "rot180": lambda a: a[::-1, ::-1], "T": lambda a: a.T, "T_flipud": lambda a: a.T[::-1],
    "T_fliplr": lambda a: a.T[:, ::-1], "T_rot180": lambda a: a.T[::-1, ::-1],
}
D8_INV = {  # NIfTI slice -> DICOM slice
    "id": lambda b: b, "flipud": lambda b: b[::-1], "fliplr": lambda b: b[:, ::-1],
    "rot180": lambda b: b[::-1, ::-1], "T": lambda b: b.T, "T_flipud": lambda b: b[::-1].T,
    "T_fliplr": lambda b: b[:, ::-1].T, "T_rot180": lambda b: b[::-1, ::-1].T,
}


def ct_series(folder: Path):
    """CT image files of ONE series in a folder (not recursive): {SeriesInstanceUID: [files]}."""
    out = {}
    files = [f for f in glob.glob(str(Path(folder) / "*")) if os.path.isfile(f)]

    def hdr(f):
        try:
            return pydicom.dcmread(f, stop_before_pixels=True,
                                   specific_tags=["Modality", "SeriesInstanceUID", "ImageType", "ImagePositionPatient"])
        except Exception:        # noqa: BLE001 -- not DICOM
            return None
    for f, h in zip(files, POOL.map(hdr, files)):
        if h is None or h.get("Modality") != "CT" or "ImagePositionPatient" not in h:
            continue             # localizer / dose-info screen captures have no slice position
        out.setdefault(str(h.SeriesInstanceUID), []).append(f)
    return out


def load_series(files):
    """Full datasets sorted along the slice normal + HU volume."""
    ds = list(POOL.map(pydicom.dcmread, files))
    iop = np.array(ds[0].ImageOrientationPatient, float)
    nrm = np.cross(iop[:3], iop[3:])
    pairs = sorted(zip(files, ds), key=lambda t: float(np.dot(nrm, np.array(t[1].ImagePositionPatient, float))))
    vol = np.empty((len(pairs), int(ds[0].Rows), int(ds[0].Columns)), np.float32)
    for k, (_, d) in enumerate(pairs):
        vol[k] = d.pixel_array * float(d.get("RescaleSlope", 1)) + float(d.get("RescaleIntercept", 0))
    return pairs, vol


def map_series(dvol: np.ndarray, nvol: np.ndarray):
    """Equal-length series: -> (NIfTI z per DICOM slice, in-plane transform). Raises unless every slice verifies."""
    if dvol.shape[0] != nvol.shape[0]:
        raise ValueError(f"slice count {dvol.shape[0]} vs {nvol.shape[0]}")
    fd, fn = dvol.std((1, 2), dtype=np.float64), nvol.std((1, 2), dtype=np.float64)
    order = np.argsort(-fd)
    found = []
    for k in order[:40]:
        z = int(np.argmin(abs(fn - fd[k])))
        for name, t in D8.items():
            tk = t(dvol[k])
            if tk.shape == nvol[z].shape:
                diff = nvol[z].astype(np.float64) - tk
                if diff.std() < 1e-3:
                    found.append((int(k), z, name, float(diff.flat[0])))
                    break
        if len(found) >= 6:
            break
    if len(found) < 2:
        raise ValueError("no pixel-identical slice pairs found")
    names = {f[2] for f in found}
    if len(names) != 1:
        raise ValueError(f"inconsistent in-plane transforms {names}")
    (k0, z0, tname, off), (k1, z1, _, _) = found[0], found[-1]
    a = (z1 - z0) / (k1 - k0) if k1 != k0 else 1
    if a not in (1, -1):
        raise ValueError(f"slice map slope {a}")
    zmap = (z0 + a * (np.arange(dvol.shape[0]) - k0)).astype(int)
    t = D8[tname]
    bad = [k for k in range(dvol.shape[0]) if np.abs(nvol[zmap[k]] - t(dvol[k]) - off).max() > 0.5]
    if bad:
        raise ValueError(f"{len(bad)} slices do not verify")
    return zmap, tname


def subset_map(dvol: np.ndarray, nvol: np.ndarray):
    """The GIPL step can drop slices: every NIfTI slice -> its DICOM twin (zmap -1 = no twin)."""
    if dvol.shape[0] < nvol.shape[0]:
        raise ValueError("series shorter than the volume")
    fd = np.round(dvol.std((1, 2), dtype=np.float64), 3)
    fn = np.round(nvol.std((1, 2), dtype=np.float64), 3)
    by_fp = {}
    for k, f in enumerate(fd):
        by_fp.setdefault(f, []).append(k)
    z0 = int(np.argmax(fn))
    tname = None
    for k in by_fp.get(fn[z0], []):
        for name, t in D8.items():
            tk = t(dvol[k])
            if tk.shape == nvol[z0].shape and np.abs(nvol[z0] - tk).max() < 0.5:
                tname = name
                break
        if tname:
            break
    if tname is None:
        raise ValueError("no pixel-identical slice")
    t = D8[tname]
    zmap = np.full(dvol.shape[0], -1, int)
    last = -1
    for z in range(nvol.shape[0]):
        ks = [k for k in by_fp.get(fn[z], []) if zmap[k] < 0 and np.abs(nvol[z] - t(dvol[k])).max() < 0.5]
        if not ks:
            raise ValueError(f"NIfTI slice {z} has no DICOM twin")
        k = min(ks, key=lambda k: abs(k - last - 1))
        zmap[k] = z
        last = k
    zs = zmap[zmap >= 0]
    if not (np.all(np.diff(zs) > 0) or np.all(np.diff(zs) < 0)):
        raise ValueError("slice order not monotone")
    return zmap, tname


def find_series_for(nvol: np.ndarray, dicom_dirs):
    """The clinical CT series this NIfTI was made from, by pixel content. -> (pairs, zmap, tname)."""
    tried = []
    for d in dicom_dirs:
        for uid, files in ct_series(d).items():
            if len(files) < nvol.shape[0]:
                continue
            pairs, dvol = load_series(files)
            try:
                zmap, tname = (map_series if len(files) == nvol.shape[0] else subset_map)(dvol, nvol)
            except ValueError as e:
                tried.append(f"{Path(d).name}/{uid[-8:]}: {e}")
                continue
            return pairs, np.asarray(zmap), tname
    raise ValueError(f"no clinical series matches the NIfTI pixel for pixel ({tried})")


# ------------------------------------------------------------------------------------------- geometry helpers
def body_mask(img: sitk.Image, anatomy: np.ndarray | None = None) -> np.ndarray:
    """Patient without couch and positioning frame. HU > -300, in-plane opening that removes thin shells, then
    the components that contain model anatomy (the TBI frame and foam box are separate from the patient);
    without anatomy: every component >= 5 % of the largest (the research version)."""
    m = sitk.BinaryMorphologicalOpening(img > -300, [4, 4, 0])
    cc = sitk.RelabelComponent(sitk.ConnectedComponent(m), sortByObjectSize=True)
    a = sitk.GetArrayFromImage(cc)
    if anatomy is not None and anatomy.any():
        ids = np.unique(a[anatomy & (a > 0)])
        sizes = np.bincount(a.ravel())
        ids = [i for i in ids if sizes[i] >= 0.001 * sizes[1:].max()]     # not a stray voxel's component
        return np.isin(a, ids)
    sizes = np.bincount(a.ravel())
    if len(sizes) < 2:
        return a > 0
    keep = np.where(sizes[1:] >= 0.05 * sizes[1])[0] + 1
    return np.isin(a, keep)


def dilate(m: np.ndarray, mm: float, sp_zyx) -> np.ndarray:
    """Binary dilation by about mm (ball kernel, radius per axis = mm / spacing)."""
    if mm <= 0 or not m.any():
        return m.copy()
    r = [max(int(round(mm / s)), 0) for s in sp_zyx[::-1]]          # sitk radius is x, y, z
    im = sitk.GetImageFromArray(m.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.BinaryDilate(im, r, sitk.sitkBall)) > 0


def erode(m: np.ndarray, mm: float, sp_zyx) -> np.ndarray:
    r = [max(int(round(mm / s)), 0) for s in sp_zyx[::-1]]
    im = sitk.GetImageFromArray(m.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.BinaryErode(im, r, sitk.sitkBall)) > 0


def close(m: np.ndarray, mm: float, sp_zyx) -> np.ndarray:
    r = [max(int(round(mm / s)), 0) for s in sp_zyx[::-1]]
    im = sitk.GetImageFromArray(m.astype(np.uint8))
    return sitk.GetArrayFromImage(sitk.BinaryMorphologicalClosing(im, r, sitk.sitkBall)) > 0


def fill_slices(m: np.ndarray) -> np.ndarray:
    out = m.copy()
    for z in np.where(m.any((1, 2)))[0]:
        out[z] = ndimage.binary_fill_holes(m[z])
    return out


def keep_main(m: np.ndarray, frac: float = 0.2) -> np.ndarray:
    """Drop stray pieces: keep 3D components at least frac x the largest (TS 'brain' had voxels in the thigh)."""
    if not m.any():
        return m
    lab, n = ndimage.label(m, ndimage.generate_binary_structure(3, 3))
    if n <= 1:
        return m
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    return np.isin(lab, np.where(sizes >= frac * sizes.max())[0])


def side_sign(ts: MaskSet, shape) -> int:
    """+1 if patient LEFT is the larger x index, -1 if the smaller: from the model's own left/right labels."""
    xl, xr, nl, nr = 0.0, 0.0, 0, 0
    for n in ts:
        if n.endswith("_left") or n.endswith("_right"):
            idx = np.nonzero(ts[n].any((0, 1)))[0]
            if not len(idx):
                continue
            if n.endswith("_left"):
                xl += idx.mean(); nl += 1
            else:
                xr += idx.mean(); nr += 1
    if not nl or not nr:
        raise ValueError("no left/right labels to orient by")
    return 1 if xl / nl > xr / nr else -1


MIDLINE = C_ + T_ + L_ + ["sacrum", "vertebrae_S1", "spinal_cord", "brain", "skull", "sternum"]


def midline_x(body: np.ndarray, axis: np.ndarray | None = None) -> np.ndarray:
    """Patient midline (x) per slice: the centroid of the axial skeleton / CNS (axis) where it is present, held
    constant beyond its ends (legs, lying straight); the body centroid only without an axis. The body alone is
    pulled sideways by frame parts touching the feet (a child's FF scan: 80-200 px instead of ~255)."""
    nz = body.shape[0]
    if axis is not None and axis.any():
        # stray labels off the axis (one cohort FF scan: 401 'skull' voxels on hands, feet and frame, z 200-370) would carry
        # the midline below the sacrum and put both legs on one side: keep the axis pieces, drop the strays
        lab, n = ndimage.label(axis, ndimage.generate_binary_structure(3, 3))
        sizes = np.bincount(lab.ravel())
        big = max(500, 0.05 * sizes[1:].max())
        axis = np.isin(lab, [i for i in range(1, n + 1) if sizes[i] >= big])
    src = axis if axis is not None and axis.sum() > 500 else body
    xc = np.full(nz, np.nan)
    xs = np.arange(body.shape[2])
    for z in range(nz):
        col = src[z].sum(0)
        if col.sum() > (50 if src is axis else 0):
            xc[z] = (col * xs).sum() / col.sum()
    ok = ~np.isnan(xc)
    if not ok.any():
        return np.full(nz, body.shape[2] / 2)
    xc = np.interp(np.arange(nz), np.nonzero(ok)[0], xc[ok])
    return np.array([np.median(xc[max(0, z - 5):z + 6]) for z in range(nz)])
