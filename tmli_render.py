"""
tmli_render.py -- one overview figure per scan and version: curved coronal (targets | OARs), sagittal, and four
axial levels with every structure outlined. For checking before import, not for the record.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from tmli_rtstruct import palette

LEVEL_HINTS = [("head", ["Eye_L", "Eye_R", "Brain"]), ("thorax", ["Heart", "Lung_L"]),
               ("abdomen", ["Kidney_L", "Kidney_R", "Liver"]), ("pelvis", ["Bladder", "CTV_M_Pelvis"]),
               ("leg", ["CTV_M_Leg_L", "CTV_M_Leg_R"])]


def _centre_z(m):
    z = np.nonzero(m.any((1, 2)))[0]
    return int(np.median(z)) if len(z) else None


def render(ct, sp, body, anat, structures, title, png: Path):
    sz, sy, sx = sp
    names = [s["name"] for s in structures]
    col = dict(zip(names, palette(len(names))))
    # view follows the structures, not the body mask: a positioning box touching the patient is part of the
    # body and would centre the coronal cut in foam (a child's FF scan)
    allm = np.zeros(ct.shape, bool)
    for s in structures:
        allm |= s["mask"]
    ref = allm if allm.any() else body
    pad = (3, 15, 15)
    zb, yb, xb = (np.nonzero(ref.any(a))[0] for a in ((1, 2), (0, 2), (0, 1)))
    z0, z1 = max(zb[0] - pad[0], 0), min(zb[-1] + pad[0] + 1, ct.shape[0])
    y0, y1 = max(yb[0] - pad[1], 0), min(yb[-1] + pad[1] + 1, ct.shape[1])
    x0, x1 = max(xb[0] - pad[2], 0), min(xb[-1] + pad[2] + 1, ct.shape[2])
    yc = np.full(ct.shape[0], np.nan)
    for z in range(ct.shape[0]):
        src = allm[z] if allm[z].sum() > 50 else body[z]
        if src.any():
            yc[z] = np.nonzero(src)[0].mean()
    ok = ~np.isnan(yc)
    yc = np.interp(np.arange(ct.shape[0]), np.nonzero(ok)[0], yc[ok])
    yc = np.array([np.median(yc[max(0, z - 6):z + 7]) for z in range(ct.shape[0])]).round().astype(int)
    zz = np.arange(ct.shape[0])
    xm = int(np.nonzero(ref.any((0, 1)))[0].mean())
    by = {s["name"]: s for s in structures}
    levels = []
    for lname, cands in LEVEL_HINTS:
        for c in cands:
            if c in by and by[c]["mask"].sum() > 50:
                z = _centre_z(by[c]["mask"])
                if all(abs(z - l) > 12 for _, l in levels):
                    levels.append((lname, z))
                break
    levels = sorted(levels, key=lambda t: -t[1])[:4]

    def lab_rgba(names_sel, take):
        rgba = np.zeros(take(ct).shape + (4,))
        for n in sorted(names_sel, key=lambda n: -int(by[n]["mask"].sum())):
            m = take(by[n]["mask"])
            rgba[m, :3] = col[n]
            rgba[m, 3] = 0.5
        return rgba

    cor = lambda v: v[zz, yc, :][z0:z1, x0:x1]
    sag = lambda v: v[z0:z1, y0:y1, xm]
    tgt = [n for n in names if by[n]["kind"] != "OAR" and by[n]["mask"].any()]
    oar = [n for n in names if by[n]["kind"] == "OAR" and by[n]["mask"].any()]
    ext = (z1 - z0) * sz
    widths = [(x1 - x0) * sx / ext] * 2 + [(y1 - y0) * sy / ext]
    h = 7.5
    fig = plt.figure(figsize=(h * sum(widths) + 1 + 4.2 * max(1, (len(levels) + 1) // 2), h + 2.6))
    gs = fig.add_gridspec(2, 3 + max(1, (len(levels) + 1) // 2), width_ratios=widths + [0.75] * max(1, (len(levels) + 1) // 2),
                          height_ratios=[1, 1], left=0.01, right=0.99, top=0.93, bottom=0.2, wspace=0.04, hspace=0.12)
    show = lambda ax, img, asp: ax.imshow(np.clip((img + 160) / 400, 0, 1), cmap="gray", origin="lower", aspect=asp)
    for c, (ttl, sel, take, asp) in enumerate((("coronal (curved): targets", tgt, cor, sz / sx),
                                                ("coronal (curved): OARs", oar, cor, sz / sx),
                                                ("sagittal: all", tgt + oar, sag, sz / sy))):
        ax = fig.add_subplot(gs[:, c])
        show(ax, take(ct), asp)
        ax.imshow(lab_rgba(sel, take), origin="lower", aspect=asp)
        if c == 0:
            for _, z in levels:
                ax.axhline(z - z0, color="y", lw=0.5, ls=":")
        ax.set_title(ttl, fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    for i, (lname, z) in enumerate(levels):
        ax = fig.add_subplot(gs[i % 2, 3 + i // 2])
        show(ax, ct[z, y0:y1, x0:x1][::-1], sy / sx)
        for n in tgt + oar:
            m = by[n]["mask"][z, y0:y1, x0:x1][::-1]
            if m.any():
                ax.contour(m, [0.5], colors=[col[n]], linewidths=0.8)
        ax.set_title(f"axial {lname} (slice {z})", fontsize=10); ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(title, fontsize=12)
    fig.legend(handles=[Patch(color=col[n], label=n) for n in tgt + oar], loc="lower center", ncol=8, fontsize=8.5,
               frameon=False)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png, dpi=110)
    plt.close(fig)
