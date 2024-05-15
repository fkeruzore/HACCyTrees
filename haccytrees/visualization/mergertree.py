import numpy as np
from collections import namedtuple
import drawsvg
import matplotlib
from typing import Mapping, Union


class _Halo(
    namedtuple(
        "_Halo",
        ["id", "mass", "freemassacc", "scale_fac", "snapnum", "depth", "progenitors"],
    )
):
    __slots__ = ()

    def __str__(self):
        return f"Halo(id={self.id}, mass={self.mass:.2e}, snapnum={self.snapnum}, depth={self.depth}, nprog={len(self.progenitors)})"

    def __repr__(self):
        return self.__str__()


def _extract_tree(
    trees, progenitor_array, target_idx, mass_threshold=1e11, max_steps=100
):
    hdict = {}
    snaplists = [[] for d in range(max_steps + 1)]
    hroot = _Halo(
        target_idx,
        trees["tree_node_mass"][target_idx],
        trees["tree_node_mass"][target_idx],
        trees["scale_factor"][target_idx],
        trees["snapnum"][target_idx],
        0,
        [],
    )
    last_step = trees["snapnum"][target_idx]
    hdict[target_idx] = hroot
    queue = [hroot]
    snaplists[0].append(hroot)
    while len(queue) > 0:
        h = queue.pop()
        prog_start = trees["progenitor_offset"][h.id]
        prog_end = prog_start + trees["progenitor_count"][h.id]
        progenitors = progenitor_array[prog_start:prog_end]
        masses = np.array([trees["tree_node_mass"][i] for i in progenitors])
        s = np.argsort(masses)[::-1]
        masses = masses[s]
        progenitors = progenitors[s]

        next_queue = []
        for m, pidx in zip(masses, progenitors):
            if m < mass_threshold:
                continue
            depth = last_step - trees["snapnum"][pidx]
            p = _Halo(
                pidx,
                m,
                m,
                trees["scale_factor"][pidx],
                trees["snapnum"][pidx],
                depth,
                [],
            )
            hdict[pidx] = p
            h.progenitors.append(p)
            h._replace(freemassacc=h.freemassacc - p.mass)
            snaplists[depth].append(p)
            if depth < max_steps:
                next_queue.append(p)
        for p in next_queue[::-1]:
            queue.append(p)
    return hdict, hroot, snaplists


def _tree_align(hlist, hloc, snaplists, padding, align="block"):
    hv = np.array([h.mass for h in hlist])
    hy = np.zeros(len(hlist), dtype=np.float64)
    snapmass = np.array([sum(h.mass for h in sl) for sl in snaplists])
    snaplen = np.array([len(sl) for sl in snaplists])

    if align == "bottom" or align == "center":
        snapvtot = snapmass * (1 + (snaplen - 1) * padding)
        snappadding = (snapvtot - snapmass) / np.clip((snaplen - 1), 1, None)
    elif align == "block":
        snapvtot = snapmass * (1 + (snaplen - 1) * padding)
        snappadding = (np.max(snapvtot) - snapmass) / np.clip((snaplen - 1), 1, None)
    else:
        raise NotImplementedError("no align {}".format(align))

    # find hy
    for i, sl in enumerate(snaplists):
        if align == "center":
            curoff = 0.5 - snapvtot[i] / 2
        else:
            curoff = 0
        for j, h in enumerate(sl):
            hy[hloc[h.id]] = curoff
            curoff += h.mass + snappadding[i]

    return hv, hy


def _tree_recursive(hroot, hlist, hloc, snaplists, padding, align):
    minpad = 0.1 * padding * hroot.mass
    maxpad = padding * hroot.mass
    padding_fct = lambda prosizes: np.clip(
        padding * (prosizes[:-1] + prosizes[1:]), minpad, maxpad
    )
    # padding_fct = lambda prosizes: padding * (prosizes[:-1] + prosizes[1:])
    # padding_fct = lambda prosizes: padding * (prosizes[:-1] + 5*prosizes[1:])/3
    # padding_fct = lambda prosizes: padding * np.sqrt(prosizes[:-1] * prosizes[1:])
    # Find vertical size of each node
    hv = np.array([h.mass for h in hlist])
    sizes = {}

    def get_size(h):
        if len(h.progenitors) == 0:
            size = h.mass
        else:
            prosizes = np.array([get_size(p) for p in h.progenitors])
            paddings = padding_fct(prosizes)
            size = max(h.mass, np.sum(prosizes) + np.sum(paddings))

        sizes[h.id] = size
        return size

    get_size(hroot)

    # position nodes (relative to root)
    vpos = {}
    if align == "center":

        def set_vpos(h, center):
            vpos[h.id] = center - h.mass / 2
            mysize = sizes[h.id]
            prosizes = np.array([sizes[p.id] for p in h.progenitors])
            paddings = padding_fct(prosizes)
            paddings = np.append(paddings, 0)
            pcenter = center - mysize / 2
            for i, p in enumerate(h.progenitors):
                pcenter += prosizes[i] / 2
                set_vpos(p, pcenter)
                pcenter += prosizes[i] / 2
                pcenter += paddings[i]

    elif align == "bottom":

        def set_vpos(h, bottom):
            vpos[h.id] = bottom
            mysize = sizes[h.id]
            prosizes = np.array([sizes[p.id] for p in h.progenitors])
            paddings = padding_fct(prosizes)
            paddings = np.append(paddings, 0)
            pbottom = bottom
            for i, p in enumerate(h.progenitors):
                set_vpos(p, pbottom)
                pbottom += prosizes[i]
                pbottom += paddings[i]

    else:
        raise NotImplementedError(f"unknown align: {align}")

    set_vpos(hroot, 0)
    hy = np.array([vpos[h.id] for h in hlist])
    return hv, hy


def merger_tree_drawing(
    trees: Mapping[str, np.ndarray],
    progenitor_array: np.ndarray,
    target_idx: int,
    *,
    max_steps: int = 20,
    mass_threshold: float = 1e10,
    method: str = "block",
    padding: float = 0.05,
    width: int = 1200,
    height: int = 600,
    cmap: Union[str, matplotlib.colors.Colormap] = "viridis",
    coloring: str = "branch",
    **kwargs,
) -> drawsvg.Drawing:
    """Visualize the merger-tree as an svg

    Parameters
    ----------
    trees
        the full merger tree forest

    progenitor_array
        the progenitor array returned by :func:`haccytrees.read_forest`

    target_idx
        the root index of the halo which is to be visualized

    max_steps
        the number of progenitor steps that are being visualized

    mass_threshold
        all progenitors below this threshold will be skipped

    method
        the drawing method that determines the y-position of each progenitor.
        See the notes for valid options

    padding
        determines the fraction of padding along the y-axis between neighboring
        progenitors

    width
        the width of the svg

    height
        the height of the svg

    cmap
        the colormap that is used to differentiate the branches

    coloring
        if ``"branch"``, will color each branch differently. If ``None``, all
        branches will be drawn in black

    kwargs
        TODO: add additional arguments

    Returns
    -------
    drawing: drawsvg.Drawing
        the svg

    Notes
    -----

    Valid ``methods`` are:

    - recursive-center
    - recursive-bottom
    - center
    - block
    - bottom
    """
    # Some config
    soft = kwargs.get("soft", 0.5)
    nodewidth = kwargs.get("nodewidth", max_steps)
    mm_threshold = kwargs.get("mm_threshold", 1 / 3)
    mm_is_absolute = kwargs.get("mm_is_absolute", False)
    highlight_mm = kwargs.get("highlight_mm", False)
    if cmap is not None and isinstance(cmap, str):
        cmap = matplotlib.pyplot.get_cmap(cmap)
    aspect = width / height

    # Get Data
    hdict, hroot, snaplists = _extract_tree(
        trees,
        progenitor_array,
        target_idx,
        mass_threshold=mass_threshold,
        max_steps=max_steps,
    )

    # Get a List of all halos
    nhalos = len(hdict)
    hlist = [h for hidx, h in hdict.items()]
    # Map from id to idx
    hloc = {h.id: i for i, h in enumerate(hlist)}

    # Determine xpos, height and width
    hnorm = max_steps * (1 + nodewidth / 100 / max_steps)
    hx = np.array([hroot.snapnum - h.snapnum for h in hlist]) / hnorm * aspect

    hh = np.array([nodewidth / 100 / max_steps for h in hlist])
    po = np.zeros(nhalos, dtype=np.float64)

    methods = method.split("-")
    print(methods, flush=True)
    if len(methods) == 1:
        methods.append("center")
    if methods[0] == "recursive":
        hv, hy = _tree_recursive(
            hroot, hlist, hloc, snaplists, padding, align=methods[1]
        )
    else:
        hv, hy = _tree_align(hlist, hloc, snaplists, padding, align=methods[0])

    # normalize to [0, 1]
    hymax = hy + hv
    voff = np.min(hy)
    vnorm = np.max(hymax) - voff
    # print(voff, vnorm)
    hy = (hy - voff) / vnorm
    hv /= vnorm

    # get links
    links = [(hloc[h.id], hloc[p.id]) for h in hlist for p in h.progenitors]
    # get progenitor offset for links
    for h in hlist:
        poff = 0
        for p in h.progenitors:
            po[hloc[p.id]] = poff
            poff += p.mass / vnorm

    if highlight_mm:
        highlight_dict = {h.id: False for h in hlist}

        def highlight_branches(h):
            highlight_dict[h.id] = True
            if len(h.progenitors):
                # highlight main progenitor
                highlight_branches(h.progenitors[0])
                abs_threshold = (
                    mm_threshold
                    if mm_is_absolute
                    else mm_threshold * h.progenitors[0].mass
                )
                for p in h.progenitors[1:]:
                    if p.mass >= abs_threshold:
                        highlight_branches(p)

        highlight_branches(hroot)
        highlights = [highlight_dict[h.id] for h in hlist]

    else:
        highlights = [True for h in hlist]

    # Plot Halos
    if coloring is None:
        colors = ["#000000" for h in hlist]
    elif coloring == "branch":
        color_dict = {}

        def set_color(h, low, up):
            color_dict[h.id] = cmap(low)
            nprogs = max(1, len(h.progenitors))
            # diff = max(up-low, 0.2)
            up = 1
            inc = (up - low) / nprogs
            for i, p in enumerate(h.progenitors):
                set_color(p, low + i * inc, low + (i + 1) * inc)

        set_color(hroot, 0, 1)
        colors = [matplotlib.colors.rgb2hex(color_dict[h.id]) for h in hlist]
    else:
        raise NotImplementedError(f"unknown coloring {coloring}")

    d = drawsvg.Drawing(aspect, 1, origin=(0, 0))
    d.setRenderSize(aspect * height, height)
    d.draw(drawsvg.Rectangle(0, 0, aspect, 1, fill="#FFF"))
    for i in range(nhalos):
        d.append(
            drawsvg.Rectangle(
                hx[i],
                hy[i],
                hh[i],
                hv[i],
                fill=colors[i],
                fill_opacity=1 if highlights[i] else 0.3,
            )
        )

    for ld, lp in links:
        _x0 = hx[ld] + hh[ld]
        _y0 = hy[ld] + po[lp]
        _h = hv[lp]
        _x1 = hx[lp]
        _y1 = hy[lp]
        _dx = _x1 - _x0
        _dy = _y1 - _y0
        _s0 = _dx * soft  # distance to control point 1
        _s1 = _dx * (1 - soft)  # distance to control point 2
        d.append(
            drawsvg.Path(
                stroke="none",
                fill=colors[lp],
                fill_opacity=0.6 if highlights[lp] else 0.1,
            )
            .M(_x0, _y0)
            .v(_h)
            .c(_s0, 0, _s1, _dy, _dx, _dy)
            .v(-_h)
            .c(-_s0, 0, -_s1, -_dy, -_dx, -_dy)
        )

    return d
