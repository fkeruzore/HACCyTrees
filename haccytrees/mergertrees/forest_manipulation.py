import numpy as np
import numba
from typing import Mapping, Union

from ..simulations import Simulation


@numba.jit(nopython=True, parallel=True)
def _get_mainbranch(snapnum, target_indices, mainbranch_matrix):
    ntargets = len(target_indices)
    nhalos = len(snapnum)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        sn = snapnum[idx]
        mainbranch_matrix[i, sn] = idx
        while idx + 1 < nhalos and snapnum[idx + 1] < sn:
            idx += 1
            sn = snapnum[idx]
            mainbranch_matrix[i, sn] = idx


def get_mainbranch_indices(
    forest: Mapping[str, np.ndarray],
    simulation: Union[str, Simulation],
    target_index: np.ndarray,
) -> np.ndarray:
    """Extract main progenitor branches in a matrix format: ``(n_targets x n_steps)``

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`

    simulation:
        either a valid simulation string or an instance of :class:`Simulation`,
        required to get the number of cosmotools steps.

    target_index:
        the indices of the halos for which the main progenitor branch should
        be extracted.

    Returns
    -------
    mainbranch_indices: np.ndarray
        the indices of the halos in the main progenitor branches. Each column `j`
        corresponds to the main branch of the `j`-th target_halo. Each row
        corresponds to a cosmotools step (with 0 being the first step). At times
        where the halo did not exist, the index is `-1`.
    """
    if isinstance(simulation, str):
        simulation = Simulation.simulations[simulation]
    if not isinstance(target_index, np.ndarray):
        if hasattr(target_index, "__len__"):
            target_index = np.array(target_index)
        else:
            target_index = np.array([target_index])
    nhalos = len(target_index)
    nsteps = len(simulation.cosmotools_steps)

    # allocate an index array
    mainbranch_indices = np.empty((nhalos, nsteps), dtype=np.int64)
    mainbranch_indices[:] = -1

    # fill index array
    _get_mainbranch(forest["snapnum"], target_index, mainbranch_indices)
    return mainbranch_indices


@numba.jit(nopython=True)
def _accumulate_infall_histogram(
    counts,
    mass,
    desc_index,
    target_index,
    branch_size,
    mass_min,
    mass_max,
):
    n_targets = len(target_index)
    nbins = len(counts)
    dm = (mass_max - mass_min) / nbins
    for j in range(n_targets):
        root_index = target_index[j]
        end_index = root_index
        for i in range(root_index + 1, root_index + branch_size[root_index]):
            if desc_index[i] == end_index:
                # main progenitor branch
                end_index = i
                continue
            if desc_index[i] > end_index:
                # doesn't merge into main branch
                continue
            mass_bin = int((mass[i] - mass_min) / dm)
            if mass_bin >= 0 and mass_bin < nbins:
                counts[mass_bin] += 1


def get_infall_histogram(
    forest: Mapping[str, np.ndarray],
    target_index: np.ndarray,
    mass_min: float,
    mass_max: float,
    nbins: int,
    logbins: bool = True,
) -> np.ndarray:
    """Get a histogram of infall masses, integrated over all snapshots and target_index

    This function counts the halos that fall onto the main progenitor branches of the
    halos specified by target_index and bins them according to their masses

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`

    target_index:
        the indices of the halos for which the infall masses should be accumulated

    mass_min:
        lower mass bound for histogram

    mass_max:
        upper mass bound for histogram

    nbins:
        number of bins in the histogram, linearly or logarithmically spread from min to
        max

    logbins:
        if ``True``, the bins will be logarithmically distributed, otherwise linearly

    Returns
    -------
    counts: np.ndarray
        the indices of the n-th most massive progenitors (determined by the
        tree-node mass). -1 if the progenitor does not exist.
    """
    counts = np.zeros(nbins, dtype=np.int64)
    mass = forest["tree_node_mass"]
    if logbins:
        mass = np.log10(mass)
        mass_min = np.log10(mass_min)
        mass_max = np.log10(mass_max)
    _accumulate_infall_histogram(
        counts,
        mass,
        forest["descendant_idx"],
        target_index,
        forest["branch_size"],
        mass_min,
        mass_max,
    )
    return counts


@numba.jit(nopython=True, parallel=True)
def _get_nth_progenitor_indices(
    progenitor_array,
    progenitor_offsets,
    progenitor_count,
    target_indices,
    progenitor_indices,
    n,
):
    ntargets = len(target_indices)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        nprogs = progenitor_count[idx]
        if nprogs < n:
            progenitor_indices[i] = -1
        else:
            progenitor_indices[i] = progenitor_array[progenitor_offsets[idx] + n - 1]


def get_nth_progenitor_indices(
    forest: Mapping[str, np.ndarray],
    progenitor_array: np.ndarray,
    target_index: np.ndarray,
    n: int,
) -> np.ndarray:
    """Extract indices of the n-th most massive progenitor for each target halo

    The index array returned has the same length as target_index. Invalid
    indices are masked with -1 (i.e. if the halo does not have n progenitors)

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`

    progenitor_array:
        the full progenitor array created by :func:`read_forest`

    target_index:
        the indices of the halos for which the merger indices should be
        extracted.

    n:
        the n-th progenitor. ``0`` corresponds to the main progenitor, ``1`` to
        the main merger halo, etc.

    Returns
    -------
    merger_indices: np.ndarray
        the indices of the n-th most massive progenitors (determined by the
        tree-node mass). -1 if the progenitor does not exist.
    """
    if n <= 0:
        raise ValueError("n needs to be larger than 0 (1==main progenitor)")

    if not isinstance(target_index, np.ndarray):
        if hasattr(target_index, "__len__"):
            target_index = np.array(target_index)
        else:
            target_index = np.array([target_index])
    nhalos = len(target_index)

    # allocate an index array
    progenitor_indices = np.empty(nhalos, dtype=np.int64)

    # fill index array
    _get_nth_progenitor_indices(
        progenitor_array,
        forest["progenitor_offset"],
        forest["progenitor_count"],
        target_index,
        progenitor_indices,
        n,
    )
    return progenitor_indices
