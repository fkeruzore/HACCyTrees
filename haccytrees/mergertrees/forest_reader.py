import numpy as np
import numba
import h5py
from typing import Tuple, Mapping, Union, Optional, List

from ..simulations import Simulation

# These fields will always be loaded from the HDF5 files
_essential_fields = ['tree_node_index', 'desc_node_index', 'snapnum', 'branch_size']

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
    snap_roots = np.empty(int(snapmax+3), dtype=np.int64)
    
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

def _create_indices(snapnum, desc_node_index):
    desc_index = _create_desc_index(snapnum, desc_node_index)
    progenitor_array, progenitor_count, progenitor_offsets = _create_progenitor_idx(desc_index)
    return desc_index, progenitor_array, progenitor_count, progenitor_offsets


def read_forest(filename: str, 
                simulation: Union[str, Simulation],
                 *,
                nchunks: int=None, 
                chunknum: int=None, 
                include_fields: List[str]=None,
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

    include_fields
        the columns that will be read from the HDF5 file. If ``None``, all data
        will be read. Note: some essential columns will always be read, check
        ``haccytrees.mergertrees.forest_reader._essential_fields``.

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
        if include_fields is None:
            include_fields = list(forest.keys())
        else:
            for k in _essential_fields:
                if not k in include_fields:
                    include_fields.append(k)
        data = {}
        for k in include_fields:
            data[k] = np.array(forest[k][start:end])

    if add_scale_factor:
        steps = np.array(simulation.cosmotools_steps)
        timestep = steps[data['snapnum']]
        data['scale_factor'] = simulation.step2a(timestep)
    
    if create_indices:
        indices = _create_indices(data['snapnum'], data['desc_node_index'])
        progenitor_array = indices[1]
        data['descendant_idx'] = indices[0]
        data['progenitor_count'] = indices[2]
        data['progenitor_offset'] = indices[3]
        data['halo_index'] = np.arange(len(indices[0]))
        return data, progenitor_array
    else:
        return data, None


@numba.jit(nopython=True, parallel=True)
def _get_mainbranch(snapnum, target_indices, mainbranch_matrix):
    ntargets = len(target_indices)
    nhalos = len(snapnum)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        sn = snapnum[idx]
        mainbranch_matrix[i, sn] = idx
        while(idx+1 < nhalos and snapnum[idx+1] < sn):
            idx += 1
            sn = snapnum[idx]
            mainbranch_matrix[i, sn] = idx


def get_mainbranch_indices(forest: Mapping[str, np.ndarray], 
                           simulation: Union[str, Simulation], 
                           target_index: np.ndarray
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
    _get_mainbranch(forest['snapnum'], target_index, mainbranch_indices)
    return mainbranch_indices


@numba.jit(nopython=True, parallel=True)
def _get_largest_merger_indices(progenitor_array, progenitor_offsets, progenitor_size, target_indices, merger_indices):
    ntargets = len(target_indices)
    for i in numba.prange(ntargets):
        idx = target_indices[i]
        nprogs = progenitor_size[idx]
        if nprogs == 0:
            merger_indices[i] = -2
        elif nprogs == 1:
            merger_indices[i] = -1
        else:
            merger_indices[i] = progenitor_array[progenitor_offsets[idx]+1]


def get_largest_merger_indices(forest: dict, 
                               progenitor_array: np.ndarray,
                               target_index: np.ndarray
    ) -> np.ndarray:
    """Extract indices of most massive merger for each target halo

    The index array returned has the same length as target_index. Invalid
    indices are masked with -1 (only one progenitor) and -2 (no progenitors)

    Parameters
    ----------
    forest: 
        the full treenode forest returned by :func:`read_forest`

    progenitor_array: 
        the full progenitor array created by :func:`read_forest`

    target_index: 
        the indices of the halos for which the merger indices should be 
        extracted.

    Returns
    -------
    merger_indices: np.ndarray the indices of the second most massive
        progenitors (determined by the tree-node mass
    """
    if not isinstance(target_index, np.ndarray):
        if hasattr(target_index, "__len__"):
            target_index = np.array(target_index)
        else:
            target_index = np.array([target_index])
    nhalos = len(target_index)

    # allocate an index array
    merger_indices = np.empty(nhalos, dtype=np.int64)

    # fill index array
    _get_largest_merger_indices(progenitor_array, forest['progenitor_offsets'], forest['progenitor_sizes'], 
                                target_index, merger_indices)
    return merger_indices


def split_fragment_tag(tag: int) -> Tuple[int, int]:
    """Extracting the original fof_tag and fragment index from fragments

    Parameters
    ----------
    tag
        the fof_tag of the fragment, has to be negative by definition

    Returns
    -------
    fof_tag
        the original fof_tag of the FoF halo the fragment is associated with
    
    fragment_idx
        the index / enumeration of the fragment (0 == main fragment)

    Notes
    -----
    This function can also be used with numpy arrays directly

    """
    tag = -tag  #reverting the - operation first
    fragment_idx = tag >> 48
    fof_tag = tag & ((1<<48) - 1)
    return fof_tag, fragment_idx
