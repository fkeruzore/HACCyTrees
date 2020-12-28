import numpy as np
import numba
import h5py
from typing import Tuple, Mapping, Union, Optional

from ..simulations import Simulation

@numba.jit(nopython=True)
def _create_desc_index(snapnum):
    N = len(snapnum)
    
    # Allocate desc_index
    desc_index = np.empty(N, dtype=np.int64)
    desc_index[:] = -1
    
    # Keep track of roots at each snap
    snapmax = np.max(snapnum)
    snap_roots = np.empty(int(snapmax+3), dtype=np.int64)
    snap_roots[:] = -1
    
    prev_sn = snapmax
    for i in range(N):
        sn = snapnum[i] + 1
        # cleanup previous step
        for j in range(sn, prev_sn):
            snap_roots[j] = -1
        desc_index[i] = snap_roots[sn+1]
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
    progenitor_array[:] = -2 # nothing should be -2 after
    
    # A temporary array to keep current offsets to write progenitor indices
    progenitor_localoffsets = np.zeros(N, dtype=np.uint32)
    
    for i in range(N):
        dix = desc_index[i]
        if dix != -1:
            write_pos = progenitor_offsets[dix] + progenitor_localoffsets[dix]
            progenitor_array[write_pos] = i
            progenitor_localoffsets[dix] += 1
        
    return progenitor_array, progenitor_count, progenitor_offsets

def _create_indices(snapnum):
    desc_index = _create_desc_index(snapnum)
    progenitor_array, progenitor_count, progenitor_offsets = _create_progenitor_idx(desc_index)
    return desc_index, progenitor_array, progenitor_count, progenitor_offsets


# new name: old name
column_rename = {
    'branch_size'     : 'branch_size', 
    'desc_node_index' : 'desc_node_index', 
    'fof_halo_count'  : 'fof_halo_count', 
    'mass_fof'        : 'fof_halo_mass', 
    'fof_halo_tag'    : 'fof_halo_tag', 
    'snap_num'        : 'snapnum', 
    'cdelta_sod'      : 'sod_halo_cdelta', 
    'cdelta_sod_error': 'sod_halo_cdelta_error', 
    'cdelta_sod_accum': 'sod_halo_c_acc_mass',
    'cdelta_sod_peak' : 'sod_halo_c_peak_mass',
    'sod_halo_count'  : 'sod_halo_count', 
    'mass_sod'        : 'sod_halo_mass', 
    'radius_sod'      : 'sod_halo_radius', 
    'tree_node_index' : 'tree_node_index', 
    'mass'            : 'tree_node_mass', 
    'xoff_com'        : 'xoff_com', 
    'xoff_fof'        : 'xoff_fof', 
    'xoff_sod'        : 'xoff_sod'
}


def read_forest(filename: str, 
                simulation: Union[str, Simulation],
                 *,
                nchunks: int=None, 
                chunknum: int=None, 
                create_indices: bool=True, 
                add_scale_factor: bool=True
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

    create_indices
        if ``True``, will add descendant_idx``, ``progenitor_count``, 
        ``progenitor_offset`` to the forest and return the ``progenitor_array``

    add_scale_factor
        if ``True``, will add the scale factor column to the forest data

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
    with h5py.File(filename, 'r') as f:
        nhalos = len(f['forest']['tree_node_index'])
        roots = np.array(f['index_499']['index'])
    nroots = len(roots)
    if nchunks is not None:
        chunksize = nroots // nchunks
        if chunknum is None:
            print("Warning: no chunknum specified, reading first chunk")
            chunknum = 0
        if chunknum >= nchunks:
            print(f"invalid chunknum: {chunknum} needs to be smaller than {nchunks}")
        start = roots[chunknum * chunksize]
        if chunknum == nchunks - 1:
            end = nhalos
        else:
            end = roots[(chunknum+1) * chunksize]
    else:
        start = 0; end = nhalos
        
    with h5py.File(filename, 'r') as f:
        forest = f['forest']
        data = {}
        for k_new, k_old in column_rename.items():
            data[k_new] = np.array(forest[k_old][start:end])

    if add_scale_factor:
        steps = np.array(simulation.cosmotools_steps)
        timestep = steps[data['snap_num']]
        data['scale_factor'] = simulation.step2a(timestep)
    
    if create_indices:
        desc_index, progenitor_array, progenitor_count, progenitor_offsets = _create_indices(data['snap_num'])
        data['descendant_idx'] = desc_index
        data['progenitor_count'] = progenitor_count
        data['progenitor_offset'] = progenitor_offsets
        data['halo_index'] = np.arange(len(desc_index))
        return data, progenitor_array
    else:
        return data, None


@numba.jit(nopython=True, parallel=True)
def _get_mainbranch(snap_num, target_indices, mainbranch_matrix):
    ntargets = len(target_indices)
    nhalos = len(snap_num)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        sn = snap_num[idx]
        mainbranch_matrix[i, sn] = idx
        while(idx+1 < nhalos and snap_num[idx+1] < sn):
            idx += 1
            sn = snap_num[idx]
            mainbranch_matrix[i, sn] = idx


def get_mainbranch_indices(forest: Mapping[str, np.ndarray], 
                           simulation: Union[str, Simulation], 
                           target_index: np.ndarray
    ) -> np.ndarray:
    """Extract main progenitor branches in a matrix format: (n_targets x n_steps)

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
    _get_mainbranch(forest['snap_num'], target_index, mainbranch_indices)
    return mainbranch_indices


@numba.jit(nopython=True, parallel=True)
def _get_largest_merger_indices(snap_num, branch_sizes, target_indices, merger_indices):
    ntargets = len(target_indices)
    nhalos = len(snap_num)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        sn = snap_num[idx]
        while(idx+1 < nhalos and snap_num[idx+1] < sn):
            idx += 1
            sn = snap_num[idx]

            merger_idx = idx + branch_sizes[idx]
            if snap_num[merger_idx] == sn:
                # it is a merger
                merger_indices[i, sn] = merger_idx
            else:
                # there is no merger
                merger_indices[i, sn] = -2


def get_largest_merger_indices(forest: dict, 
                               simulation: Union[str, Simulation], 
                               target_index: np.ndarray
    ) -> np.ndarray:
    """Extract indices of most massive merger at every timestep

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`
    
    simulation:
        either a valid simulation string or an instance of :class:`Simulation`,
        required to get the number of cosmotools steps.

    target_index:
        the indices of the halos for which the merger indices should be 
        extracted.

    Returns
    -------
    merger_indices: np.ndarray
        the indices of the halos that are. Each column `j`
        corresponds to the main branch of the `j`-th target_halo. Each row
        corresponds to a cosmotools step (with 0 being the first step). At times
        where the halo did not exist, the index is `-1`. If there is no merger
        at that time (i.e. only one progenitor), the index is `-2`
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
    merger_indices = np.empty((nhalos, nsteps), dtype=np.int64)
    merger_indices[:] = -1

    # fill index array
    _get_largest_merger_indices(forest['snap_num'], forest['branch_size'], target_index, merger_indices)
    return merger_indices


def get_merger_ratio(forest: dict, 
                     simulation: Union[str, Simulation], 
                     target_index: np.ndarray
    ) -> np.ndarray:
    """Extract merger ratios

    Parameters
    ----------
    forest:
        the full treenode forest returned by :func:`read_forest`
    
    simulation:
        either a valid simulation string or an instance of :class:`Simulation`,
        required to get the number of cosmotools steps.

    target_index:
        the indices of the halos along whose main progenitor branch the merger
        ratio should be calculated

    Returns
    -------
    merger_ratios: np.ndarray
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
    merger_indices = np.empty((nhalos, nsteps), dtype=np.int64)
    merger_indices[:] = -1

    # fill index array
    _get_largest_merger_indices(forest['snap_num'], forest['branch_size'], target_index, merger_indices)
    return merger_indices


def split_fragment_tag(tag: int) -> Tuple[int, int]:
    tag = -tag  #reverting the - operation first
    idx = tag >> 48
    old_tag = tag & ((1<<48) - 1)
    return old_tag, idx
