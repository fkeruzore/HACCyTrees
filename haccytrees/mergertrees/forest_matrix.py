import numpy as np
import numba
from typing import Mapping, Union

from ..simulations import Simulation


@numba.jit(nopython=True)
def _count_rows(snapnum, desc_index, mass, row_idx, mass_threshold=0):
    nhalos = len(snapnum)
    lastsnap = snapnum[0]
    current_row = 0
    row_above_threshold = False
    for i in range(nhalos):
        if desc_index[i] == -1 or snapnum[i] >= lastsnap:
            row_above_threshold = mass[i] > mass_threshold
            if row_above_threshold:
                current_row += 1
        # elif snapnum[i] >= lastsnap:
        #     current_row += 1
        lastsnap = snapnum[i]
        row_idx[i] = current_row - 1 if row_above_threshold else -1
    return current_row


@numba.jit(nopython=True, parallel=True)
def _fill_matrix(mat, data, row_idx, col_idx):
    for i in numba.prange(len(data)):
        if row_idx[i] > -1:
            mat[row_idx[i], col_idx[i]] = data[i]


@numba.jit(nopython=True)
def _fill_hostidx(tree_node_index_mat, top_host_row_mat, direct_host_row_mat):
    nrows = tree_node_index_mat.shape[0]
    ncols = tree_node_index_mat.shape[1]
    host_rows_per_snap = np.empty(ncols, dtype=np.int64)
    host_rows_per_snap[:] = -1
    for i in range(nrows):
        infall_col = 0
        for j in range(ncols):
            if tree_node_index_mat[i, j] == 0:
                if infall_col == 0:
                    # halo doesn't exist yet at this snapshot
                    continue
                else:
                    # this is a subhalo
                    host_row = host_rows_per_snap[j]
                    top_host_row = top_host_row_mat[host_row, j]
                    direct_host_row_mat[i, j] = host_row
                    top_host_row_mat[i, j] = (
                        top_host_row if top_host_row != -1 else host_row
                    )
            else:
                # a top halo
                infall_col = j
                host_rows_per_snap[j] = i


def forest2matrix(
    forest: Mapping[str, np.ndarray],
    simulation: Union[str, Simulation],
    target_index: int = None,
    *,
    subhalo_data: Mapping[str, np.ndarray] = None,
    branchmass_threshold: float = None
) -> Mapping[str, np.ndarray]:
    """Convert a haccytree forest to a matrix, where each row is a branch

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`

    simulation:
        the full progenitor array created by :func:`read_forest`

    target_index:
        (optional) if set, a matrix of only the tree starting at `target_index` will be
        calculated

    subhalo_data:
        currently does nothing

    branchmass_threshold:
        if set, removes all branches that have a mass below the threshold at infall


    Returns
    -------
    matrices: Mapping[str, np.ndarray]
        arrays of shape `(nbranches, nsnapshots)` for each of the forest properties.
        Additionally, `matrices["top_host_row"]` contains the row number of the main
        host, and `matrices["direct_host_row"]` contains the row of the direct host in
        the hierarchy. Both are `-1` for host halos


    Notes
    -----
    All the properties (except `top_host_row` and `direct_host_row`) are 0 for entries
    where the halo does not exist or when it's a subhalo.

    """
    if isinstance(simulation, str):
        simulation = Simulation.simulations[simulation]

    if target_index is not None:
        start = target_index
        end = target_index + forest["branch_size"][target_index]
        forest = {k: forest[k][start:end] for k in forest.keys()}
        # make sure we don't use invalid indices...
        forest.pop("halo_index", None)
        forest.pop("descendant_idx", None)

    ncols = len(simulation.cosmotools_steps)
    nhalos = len(forest["snapnum"])
    col_idx = forest["snapnum"]

    row_idx = np.empty(nhalos, dtype=np.int64)
    mass_threshold = 0 if branchmass_threshold is None else branchmass_threshold
    nrows = _count_rows(
        forest["snapnum"],
        forest["desc_node_index"],
        forest["tree_node_mass"],
        row_idx,
        mass_threshold=mass_threshold,
    )

    # remove forest specific keys
    data_keys = set(forest.keys())
    discard_keys = [
        "branch_size",
        "descendant_idx",
        "progenitor_count",
        "progenitor_offset",
        "halo_index",
        "scale_factor",
        "snapnum",
    ]
    for k in discard_keys:
        data_keys.discard(k)

    matrices = {}
    for k in data_keys:
        data = forest[k]
        matrices[k] = np.zeros((nrows, ncols), dtype=data.dtype)
        _fill_matrix(matrices[k], data, row_idx, col_idx)

    # subhalo data: contains 'mass', 'hostidx', 'direct_hostidx', 'infallidx', 'snapnum'
    if subhalo_data is not None:
        # copy mass
        pass

    # Additional indices
    matrices["top_host_row"] = np.empty((nrows, ncols), dtype=np.int64)
    matrices["top_host_row"][:] = -1
    matrices["direct_host_row"] = np.empty((nrows, ncols), dtype=np.int64)
    matrices["direct_host_row"][:] = -1
    _fill_hostidx(
        matrices["tree_node_index"],
        matrices["top_host_row"],
        matrices["direct_host_row"],
    )

    return matrices
