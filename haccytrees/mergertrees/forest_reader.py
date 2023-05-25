import numpy as np
import numba
import h5py
from typing import Tuple, Mapping, Union, Optional, List

from ..simulations import Simulation

# These fields will always be loaded from the HDF5 files
_essential_fields = ["tree_node_index", "desc_node_index", "snapnum", "branch_size"]


@numba.jit(nopython=True)
def _create_desc_index(snapnum, desc_node_index):
    """Creating an array index for each halo, given by the data order

    Parameters
    ----------
    snapnum
        the full "snapnum" data array of the merger forest

    desc_node_index
        the desc_node_index array of the merger forests, required since we have
        some trees that are not rooted at z=0 (final snapshot), and therefore
        we need to know when we start a new tree (desc_node_index==-1)

    Notes
    -----
    The function keeps track of the current tree hierachy by the snap_roots
    array, where each entry has the array index of the last halo at that
    snapshot as we traverse a tree. The descendant index of a halo at index i is
    then the value of ``snap_roots[snapnum[i] + 1]``

    """
    N = len(snapnum)

    # Allocate desc_index
    desc_index = np.empty(N, dtype=np.int64)
    desc_index[:] = -1

    # Keep track of roots at each snap
    snapmax = np.max(snapnum)
    snap_roots = np.empty(int(snapmax + 3), dtype=np.int64)
    snap_roots[:] = -1

    prev_sn = snapmax
    for i in range(N):
        sn = snapnum[i] + 1
        # check if it's a new root
        if desc_node_index[i] < 0:
            # this is only necessary because we may have roots at z>0 (halos
            # that disappear)
            snap_roots[:] = -1
        else:
            # cleanup previous step
            for j in range(sn, prev_sn):
                snap_roots[j] = -1

        desc_index[i] = snap_roots[sn + 1]
        snap_roots[sn] = i
        prev_sn = sn

    return desc_index


@numba.jit(nopython=True)
def _create_progenitor_idx(desc_index):
    N = len(desc_index)

    # Count how many progenitors each halo has
    progenitor_count = np.zeros(N, dtype=np.uint32)
    for i in range(N):
        dix = desc_index[i]
        if dix != -1:
            progenitor_count[dix] += 1

    progenitor_offsets = np.empty(N, dtype=np.uint64)
    progenitor_array_size = 0
    for i in range(N):
        progenitor_offsets[i] = progenitor_array_size
        progenitor_array_size += progenitor_count[i]

    # Create an array that will hold indices to progenitors for each halo
    progenitor_array = np.empty(progenitor_array_size, dtype=np.int64)
    progenitor_array[:] = -2  # nothing should be -2 after

    # A temporary array to keep current offsets to write progenitor indices
    progenitor_localoffsets = np.zeros(N, dtype=np.uint32)

    for i in range(N):
        dix = desc_index[i]
        if dix != -1:
            write_pos = progenitor_offsets[dix] + progenitor_localoffsets[dix]
            progenitor_array[write_pos] = i
            progenitor_localoffsets[dix] += 1

    return progenitor_array, progenitor_count, progenitor_offsets


def _create_indices(snapnum, desc_node_index):
    desc_index = _create_desc_index(snapnum, desc_node_index)
    progenitor_array, progenitor_count, progenitor_offsets = _create_progenitor_idx(
        desc_index
    )
    return desc_index, progenitor_array, progenitor_count, progenitor_offsets


@numba.jit(nopython=True)
def _get_massthreshold_mask(
    snapnum, tree_node_mass, desc_node_index, mask, threshold_mass, snap0
):
    nhalos = len(snapnum)
    snap_roots = np.empty(snap0 + 1, dtype=numba.bool_)
    snap_roots[:] = True
    lastsnap = snapnum[0]
    for i in range(nhalos):
        # print(i, snap_roots)
        if desc_node_index[i] < 0:
            # a completely new root
            snap_roots[:] = True

        elif snapnum[i] >= lastsnap:
            # we are at a new branch
            for s in range(lastsnap, snapnum[i]):
                snap_roots[s] = True
        desc_valid = desc_node_index[i] < 0 or snap_roots[snapnum[i] + 1]
        mask[i] = tree_node_mass[i] > threshold_mass and desc_valid
        snap_roots[snapnum[i]] = mask[i]
        lastsnap = snapnum[i]


@numba.jit(nopython=True, parallel=True)
def _fix_branch_size(branch_size, mask):
    nhalos = len(branch_size)
    cumulative_counter = np.zeros(nhalos + 1, dtype=np.uint64)
    current_counter = 0
    for i in range(nhalos):
        cumulative_counter[i] = current_counter
        current_counter += mask[i]
    cumulative_counter[nhalos] = current_counter

    for i in numba.prange(nhalos):
        end = i + branch_size[i]
        branch_size[i] = cumulative_counter[end] - cumulative_counter[i]


def read_forest(
    filename: str,
    simulation: Union[str, Simulation],
    *,
    nchunks: int = None,
    chunknum: int = None,
    include_fields: List[str] = None,
    create_indices: bool = True,
    add_scale_factor: bool = True,
    mass_threshold: float = None,
    include_non_z0_roots: bool = False,
) -> Tuple[Mapping[str, np.ndarray], Optional[np.ndarray]]:
    """Read a HDF5 merger-forest

    Parameters
    ----------

    filename
        the path to the merger forest file

    simulation
        either a valid simulation name or a Simulation instance, used to add the
        scale factor to the output

    nchunks
        if not None, the file will be split in ``nchunks`` equally-sized parts

    chunknum
        if ``nchunks`` is set, ``chunknum`` determines which chunk number will
        be read. First chunk has ``chunknum=0``, has to be smaller than
        ``nchunks``.

    include_fields
        the columns that will be read from the HDF5 file. If ``None``, all data
        will be read. Note: some essential columns will always be read, check
        ``haccytrees.mergertrees.forest_reader._essential_fields``.

    create_indices
        if ``True``, will add descendant_idx``, ``progenitor_count``,
        ``progenitor_offset`` to the forest and return the ``progenitor_array``

    add_scale_factor
        if ``True``, will add the scale factor column to the forest data

    mass_threshold
        if not ``None``, the reader will prune all halos below the specified
        mass threshold (in Msun/h)

    include_non_z0_roots
        if True, will also include trees that are not rooted at z=0 (i.e. halos
        that for some reason "disappear" during the evolution)

    Returns
    -------
    forest: Mapping[str, np.ndarray]
        the merger tree forest data

    progenitor_array: Optional[np.ndarray]
        a progenitor index array that can be used together with the
        ``progenitor_offset`` and ``progenitor_count`` arrays in the forest
        data in order to easily find all progenitors of a halo. Only returned if
        ``create_indices=True``.

    """
    if isinstance(simulation, str):
        if simulation[:-4] == ".cfg":
            simulation = Simulation.parse_config(simulation)
        else:
            simulation = Simulation.simulations[simulation]

    root_step = simulation.cosmotools_steps[-1]

    with h5py.File(filename, "r") as f:
        nhalos = len(f["forest"]["tree_node_index"])
        roots = np.array(f[f"index_{root_step}"]["index"])
        nroots = len(roots)
        if include_non_z0_roots:
            file_end = nhalos
        else:
            file_end = roots[-1] + f["forest"]["branch_size"][roots[-1]]

    if nchunks is not None:
        chunksize = nroots // nchunks
        if chunknum is None:
            print("Warning: no chunknum specified, reading first chunk")
            chunknum = 0
        if chunknum >= nchunks:
            print(f"invalid chunknum: {chunknum} needs to be smaller than {nchunks}")
        start = roots[chunknum * chunksize]
        if chunknum == nchunks - 1:
            end = file_end
        else:
            end = roots[(chunknum + 1) * chunksize]
    else:
        start = 0
        end = file_end

    with h5py.File(filename, "r") as f:
        forest = f["forest"]
        if include_fields is None:
            include_fields = list(forest.keys())
        else:
            for k in _essential_fields:
                if k not in include_fields:
                    include_fields.append(k)
        data = {}
        for k in include_fields:
            data[k] = np.array(forest[k][start:end])

    if mass_threshold is not None:
        mask = np.empty(len(data["snapnum"]), dtype=np.bool)
        # snapnum, tree_node_mass, desc_node_index, mask, threshold_mass, snap0
        _get_massthreshold_mask(
            data["snapnum"],
            data["tree_node_mass"],
            data["desc_node_index"],
            mask,
            mass_threshold,
            len(simulation.cosmotools_steps),
        )

        # Fix branch size
        _fix_branch_size(data["branch_size"], mask)

        # Apply mask
        for k in data.keys():
            data[k] = data[k][mask]

    if add_scale_factor:
        steps = np.array(simulation.cosmotools_steps)
        timestep = steps[data["snapnum"]]
        data["scale_factor"] = simulation.step2a(timestep)

    if create_indices:
        indices = _create_indices(data["snapnum"], data["desc_node_index"])
        progenitor_array = indices[1]
        data["descendant_idx"] = indices[0]
        data["progenitor_count"] = indices[2]
        data["progenitor_offset"] = indices[3]
        data["halo_index"] = np.arange(len(indices[0]))
        return data, progenitor_array
    else:
        return data, None


def read_forest_targets(
    filename: str,
    simulation: Union[str, Simulation],
    root_indices: List[int],
    *,
    include_fields: List[str] = None,
    create_indices: bool = True,
    add_scale_factor: bool = True,
    mass_threshold: float = None,
) -> Tuple[Mapping[str, np.ndarray], Optional[np.ndarray]]:
    """Read a HDF5 merger-forest

    Parameters
    ----------

    filename
        the path to the merger forest file

    simulation
        either a valid simulation name or a Simulation instance, used to add the
        scale factor to the output

    root_indices
        the start points of the subtrees that should be read (offset in the forest
        arrays)

    include_fields
        the columns that will be read from the HDF5 file. If ``None``, all data
        will be read. Note: some essential columns will always be read, check
        ``haccytrees.mergertrees.forest_reader._essential_fields``.

    create_indices
        if ``True``, will add descendant_idx``, ``progenitor_count``,
        ``progenitor_offset`` to the forest and return the ``progenitor_array``

    add_scale_factor
        if ``True``, will add the scale factor column to the forest data

    mass_threshold
        if not ``None``, the reader will prune all halos below the specified
        mass threshold (in Msun/h)

    Returns
    -------
    forest: Mapping[str, np.ndarray]
        the merger tree forest data

    progenitor_array: Optional[np.ndarray]
        a progenitor index array that can be used together with the
        ``progenitor_offset`` and ``progenitor_count`` arrays in the forest
        data in order to easily find all progenitors of a halo. Only returned if
        ``create_indices=True``.

    """
    if isinstance(simulation, str):
        simulation = Simulation.simulations[simulation]
    with h5py.File(filename, "r") as f:
        roots = np.array(f["index_499"]["index"])
        file_end = roots[-1] + f["forest"]["branch_size"][roots[-1]]
        assert np.max(root_indices) < file_end
        starts = np.array(root_indices)
        ends = starts + f["forest"]["branch_size"][root_indices]

    root_sizes = ends - starts
    data_size = np.sum(root_sizes)

    with h5py.File(filename, "r") as f:
        forest = f["forest"]
        if include_fields is None:
            include_fields = list(forest.keys())
        else:
            for k in _essential_fields:
                if k not in include_fields:
                    include_fields.append(k)

        # allocate
        data = {}
        for k in include_fields:
            data[k] = np.empty(data_size, dtype=forest[k].dtype)

        # read
        offset = 0
        for i in range(len(root_indices)):
            for k in include_fields:
                data[k][offset : offset + root_sizes[i]] = forest[k][
                    starts[i] : ends[i]
                ]
            offset += root_sizes[i]
        assert offset == data_size

    if mass_threshold is not None:
        mask = np.empty(len(data["snapnum"]), dtype=np.bool)
        # snapnum, tree_node_mass, desc_node_index, mask, threshold_mass, snap0
        _get_massthreshold_mask(
            data["snapnum"],
            data["tree_node_mass"],
            data["desc_node_index"],
            mask,
            mass_threshold,
            len(simulation.cosmotools_steps),
        )

        # Fix branch size
        _fix_branch_size(data["branch_size"], mask)

        # Apply mask
        for k in data.keys():
            data[k] = data[k][mask]

    if add_scale_factor:
        steps = np.array(simulation.cosmotools_steps)
        timestep = steps[data["snapnum"]]
        data["scale_factor"] = simulation.step2a(timestep)

    if create_indices:
        indices = _create_indices(data["snapnum"], data["desc_node_index"])
        progenitor_array = indices[1]
        data["descendant_idx"] = indices[0]
        data["progenitor_count"] = indices[2]
        data["progenitor_offset"] = indices[3]
        data["halo_index"] = np.arange(len(indices[0]))
        return data, progenitor_array
    else:
        return data, None
