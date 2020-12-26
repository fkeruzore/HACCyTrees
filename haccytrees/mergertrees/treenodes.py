from typing import Callable, Sequence, Mapping, Tuple
import numpy as np
import pygio
import numba
import gc, sys

from ..simulations import _Simulation
from ..utils.partition import Partition
from ..utils.distribute import distribute, exchange
from ..utils.timer import Timer, time
from ..utils.datastores import GenericIOStore, NumpyStore
from ..utils.memory import debug_memory
from .treenodes_output_hdf5 import write2multiple, write2single


# Required columns to run the algorithm
_fields_essential = [
    "tree_node_index",
    "desc_node_index",
    'fof_halo_center_x',
    'fof_halo_center_y',
    'fof_halo_center_z',
    "tree_node_mass",
]

# Coordinates that will be used to assign halos initially
_fields_xyz = ['fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z']


@numba.jit(nopython=True)
def _descendant_index(ids, desc_ids):
    desc_idx = np.copy(desc_ids)
    ids2idx = numba.typed.Dict.empty(numba.int64, numba.int64)
    for i in range(len(ids)):
        ids2idx[ids[i]] = i

    progenitor_sizes = np.zeros_like(ids, dtype=np.uint32)

    for i in range(len(desc_ids)):
        if desc_ids[i] != -1:
            dix = ids2idx[desc_ids[i]]
            desc_idx[i] = dix
            progenitor_sizes[dix] += 1

    progenitor_offsets = np.empty_like(progenitor_sizes)

    progenitor_array_size = 0
    for i in range(len(ids)):
        progenitor_offsets[i] = progenitor_array_size
        progenitor_array_size += progenitor_sizes[i]

    progenitor_internal_offsets = np.zeros_like(progenitor_offsets)
    progenitor_array = np.empty(progenitor_array_size, dtype=np.uint32)

    for i in range(len(desc_ids)):
        if desc_ids[i] != -1:
            dix = ids2idx[desc_ids[i]]
            progenitor_array[progenitor_offsets[dix] + progenitor_internal_offsets[dix]] = i
            progenitor_internal_offsets[dix] += 1

    return desc_idx, progenitor_array, progenitor_offsets


@numba.jit(nopython=True)
def _fill_branch_pos_desc(branch_position, parent_branch_position, branch_size, progenitor_array, progenitor_offsets):
    for j in range(len(parent_branch_position)):
        current_pos = parent_branch_position[j] + 1
        low = progenitor_offsets[j]
        up = progenitor_offsets[j+1] if j+1 < len(progenitor_offsets) else len(progenitor_array)
        for pj in range(low, up):
            p = progenitor_array[pj]
            branch_position[p] = current_pos
            current_pos += branch_size[p]


@numba.jit(nopython=True)
def _fill_branch_pos_new(branch_position, branch_size, current_pos, desc_index):
    for j in range(len(desc_index)):
        if desc_index[j] == -1:
            branch_position[j] = current_pos
            current_pos += branch_size[j]
        else:
            assert(branch_position[j] != -1)
    return current_pos


@numba.jit(nopython=True)
def _calculate_branch_size(branch_size, prev_branch_size, desc_index):
    for j in range(len(desc_index)):
        if desc_index[j] != -1:
            branch_size[desc_index[j]] += prev_branch_size[j]


def _fill_branch_pos(branch_position, parent_branch_position, branch_size, desc_index, progenitor_array, progenitor_offsets, new_branch_insert_pos):
    # loop over all progenitors
    if len(parent_branch_position) > 0:
        _fill_branch_pos_desc(branch_position, parent_branch_position, branch_size, progenitor_array, progenitor_offsets)
    
    # loop over all with desc == -1:
    current_pos = _fill_branch_pos_new(branch_position, branch_size, new_branch_insert_pos, desc_index)
    return current_pos


def _read_data(partition, simulation, filename, logger, fields_copy, fields_derived, rebalance, verbose):
    # List of elements that needs to be read
    fields_read = _fields_essential + fields_copy + sum([f[0] for k, f in fields_derived.items()], [])
    # make unique (need to sort, otherwise ordering not deterministic, which screws up MPI)
    fields_read = sorted(list(set(fields_read)))

    with Timer(name="read treenodes: read genericio", logger=logger):
        data = pygio.read_genericio(filename, fields_read, 
            method=pygio.PyGenericIO.FileIO.FileIOMPI,
            rebalance_sourceranks=rebalance
            )

    # normalize data
    for k in _fields_xyz:
        data[k] = np.remainder(data[k], simulation.rl)

    # assign to rank by position
    with Timer(name="read treenodes: distribute", logger=logger):
        data = distribute(partition, data, _fields_xyz, verbose=verbose, verify_count=True)
    
    # calculate derived fields
    with Timer(name="read treenodes: calculate derived fields", logger=logger):
        for k, f in fields_derived.items():
            data[k] = f[1](data, simulation)

    # only keep necessary fields
    data_new = {}
    keys = sorted(list(set(_fields_essential + fields_copy + list(fields_derived.keys()))))
    for k in keys:
        data_new[k] = data[k]

    return data_new    


def _catalog2tree_step(step: int, data_by_step, index_by_step, size_by_step, local_desc, partition: Partition, simulation: _Simulation, treenode_base: str, fields_copy: list, fields_derived: dict,
                 do_all2all_exchange, fail_on_desc_not_found, rebalance_gio_read, verbose, logger):
    logger(f'\nSTEP {step:3d}\n--------\n')
        
    # read data and calculate derived fields (contains its own timers)
    with Timer(name="read treenodes", logger=logger):
        data = _read_data(partition, simulation, f"{treenode_base}-{step}.treenodes", logger, fields_copy, fields_derived, rebalance=rebalance_gio_read, verbose=verbose)

    # exchange progenitors that belong to neighboring ranks
    with Timer(name="exchange descendants", logger=logger):
        if fail_on_desc_not_found:
            na_desc_key = None
        else:
            na_desc_key = -1
        data = exchange(partition, data, 'desc_node_index', local_desc, filter_key=lambda idx: idx >= 0, verbose=verbose, do_all2all=do_all2all_exchange, replace_notfound_key=na_desc_key)

    # sort array by descendant index, most massive first
    with Timer(name="sort arrays", logger=logger):
        s = np.lexsort((data['desc_node_index'], data['tree_node_mass']))[::-1]
        for k in data.keys():
            data[k] = data[k][s]

    # get index to descendant for each progenitor, and an inverse list of progenitors for each descendant
    with Timer(name="assign progenitors", logger=logger):
        desc_idx, progenitor_array, progenitor_offsets = _descendant_index(local_desc, data['desc_node_index'])
        if progenitor_array.min() < 0:
            print(f"invalid progenitor array on rank {partition.rank}", file=sys.stderr, flush=True)
            partition.comm.Abort()
    # prepare next step:
    local_ids = data['tree_node_index']

    # store data
    data_by_step[step] = data
    index_by_step[step] = {'desc_idx': desc_idx, 'progenitor_array': progenitor_array, 'progenitor_offsets': progenitor_offsets}
    size_by_step[step] = len(local_ids)
    
    return local_ids


def catalog2tree(simulation: _Simulation, 
                 treenode_base: str, 
                 fields_copy: Sequence[str], 
                 fields_derived: Mapping[str, Tuple[Sequence[str], Callable]], 
                 output_file: str, 
                 *,  # The following arguments are keyword-only
                 temporary_path: str=None, 
                 do_all2all_exchange: bool=False, 
                 split_output: bool=False, 
                 fail_on_desc_not_found: bool=True,
                 rebalance_gio_read: bool=False, 
                 mpi_waittime: float=0, 
                 logger: Callable[[str],None]=None, 
                 verbose: bool=False
                 ) -> None:
    """The main function that converts treenode-catalogs to treenode forests

    [add basic outline of algorithm]

    Parameters
    ----------
    
    simulation: _Simulation

    treenode_base: str

    fields_copy: Sequenc

    """
    # Set a timer for the full run
    total_timer = Timer("total time", logger=None)
    total_timer.start()
    
    # Initialize MPI partition
    time.sleep(mpi_waittime)
    partition = Partition(simulation.rl, 
        create_topo26=not do_all2all_exchange, 
        mpi_waittime=mpi_waittime)

    # Wait a sec... (maybe solves MPI issues?)
    time.sleep(mpi_waittime)

    # Used to print status messages, only from rank 0
    def logger(x, **kwargs):
        kwargs.pop("flush", None)
        partition.comm.Barrier()
        if partition.rank == 0:
            print(x, flush=True, **kwargs)
    
    # Print MPI configuration
    if partition.rank == 0:
        print(f"Running catalog2tree with {partition.nranks} ranks")
        if verbose:
            print("Partition topology:")
            print("  Decomposition: ", partition.decomp)
            print("  Ranklist: \n", partition.ranklist)
            print("", flush=True)
    if verbose > 1:
        for i in range(partition.nranks):
            if partition.rank == i:
                print(f"Rank {i}", flush=True)
                print("  Coordinates: ", partition.coordinates)
                print("  Extent:      ", partition.extent)
                print("  Origin:      ", partition.origin)
                print("  Neighbors:   ", partition.neighbors)
                if not do_all2all_exchange:
                    n26 = partition.neighbors26
                    print("  26-neighbors N : ", len(n26))
                    print("  26-neighbors   : ", n26)
                print("", flush=True)
            partition.comm.Barrier()


    # cosmotools steps
    steps = simulation.cosmotools_steps

    # read final snapshot (tree roots)
    with Timer(name="read treenodes", logger=logger):
        data = _read_data(partition, simulation, f"{treenode_base}-{steps[-1]}.treenodes", logger, fields_copy, fields_derived, rebalance=rebalance_gio_read, verbose=verbose)

    # sort by tree_node_index
    with Timer(name="sort arrays", logger=logger):
        s = np.argsort(data['tree_node_index'])
        for k in data.keys():
            data[k] = data[k][s]

    local_ids = data['tree_node_index']
    dtypes = {k: i.dtype for k, i in data.items()}

    # Containers to store local data from each snapshot
    data_by_step = GenericIOStore(partition, temporary_path)
    data_by_step[steps[-1]] = data

    index_by_step = NumpyStore(partition, temporary_path)
    index_by_step[steps[-1]] = {
        'desc_idx': -1.*np.ones(len(s), dtype=np.int64), 
        'progenitor_array': np.empty(0, dtype=np.uint32), 
        'progenitor_offsets': np.empty(0, dtype=np.uint32)}
    size_by_step = {steps[-1]: len(local_ids)}

    # iterate over previous snapshots: read, assign, prepare ordering
    for step in steps[-2::-1]:
        local_ids = _catalog2tree_step(step, data_by_step, index_by_step, size_by_step, local_ids, partition, simulation, treenode_base, fields_copy, fields_derived, do_all2all_exchange, fail_on_desc_not_found, rebalance_gio_read, verbose, logger) 
        gc.collect()  # explicitly free memory
        if verbose:
            debug_memory(partition, "AFTER GC")

    # local and total size of forest
    local_size = sum([l for s, l in size_by_step.items()])

    logger("\nBUILDING TREE\n-------------\n")
    # size of branches, iteratively
    with Timer("calculate branch sizes", logger=logger):
        branch_sizes = {step: np.ones(l, dtype=np.uint64) for step, l in size_by_step.items()}
        desc_index = np.empty(0, dtype=np.int64)
        prev_branch_size = np.empty(0, dtype=np.uint64)
        for i, step in enumerate(steps):
            branch_size = branch_sizes[step]
            _calculate_branch_size(branch_size, prev_branch_size, desc_index)
            
            prev_branch_size = branch_size
            next_indices = index_by_step[step]
            desc_index = next_indices['desc_idx']

    if verbose:
        debug_memory(partition, "AFTER BRANCH SIZE")

    # index to where in the forest each halo goes
    with Timer("calculate array positions", logger=logger):
        branch_positions = {}
        parent_branch_position = np.empty(0, dtype=np.int64)
        new_branch_insert_pos = 0
        for i, step in enumerate(steps[::-1]):
            branch_size = branch_sizes[step]
            next_indices = index_by_step[step]
            desc_index = next_indices['desc_idx']
            # convert array and offsets to list of lists
            # progenitor_idx = np.array_split(next_indices['progenitor_array'], next_indices['progenitor_offsets'][1:])

            branch_position = -np.ones(len(branch_size), dtype=np.int64)
            current_pos = _fill_branch_pos(branch_position, parent_branch_position, branch_size, desc_index, next_indices['progenitor_array'], next_indices['progenitor_offsets'], new_branch_insert_pos)
            # mark end-of-table
            new_branch_insert_pos = current_pos

            # update for next iteration
            parent_branch_position = branch_position
            branch_positions[step] = branch_position

    if verbose:
        debug_memory(partition, "AFTER ARRAY POS")

    # Assert that we got all halos
    if(new_branch_insert_pos != local_size):
        print(f"Error (rank {partition.rank}): forest size is {new_branch_insert_pos} instead of {local_size}", flush=True)
        partition.comm.Abort()

    # write data
    with Timer("output HDF5 forest", logger=logger):
        fields_write = fields_copy + list(fields_derived.keys())
        if split_output:
            write2multiple(partition, steps, local_size, branch_sizes, branch_positions, data_by_step, fields_write, dtypes, output_file, logger)
        else:
            write2single(partition, steps, local_size, branch_sizes, branch_positions, data_by_step, fields_write, dtypes, output_file, logger)

    logger("cleanup...")
    for step in simulation.cosmotools_steps:
        data_by_step.remove(step)
        index_by_step.remove(step)

    total_timer.stop()

    if partition.rank == 0:
        print("\n----------------------------------------\n")
        print("Forest creation finished!\n")
        print("Final timers:")
        timer_names = list(Timer.timers.keys())
        timer_names.sort()
        maxtimerlen = max([len(n) for n in timer_names]) + 1
        for n, t in Timer.timers.items():
            print(f"{n:{maxtimerlen}s}: {t:4e}s")

    partition.comm.Barrier()
