from typing import Callable, Mapping, Union
import numpy as np
import pygio
import numba
import gc
import sys
from mpi4py import MPI

from ...simulations import Simulation
from mpipartition import Partition
from mpipartition import distribute, exchange
from ...utils.timer import Timer, time
from ...utils.datastores import GenericIOStore, NumpyStore
from ...utils.memory import debug_memory
from .catalogs2trees_hdf5_output import write2multiple, write2single
from .fieldsconfig import FieldsConfig


@numba.jit(nopython=True)
def _descendant_index(ids, desc_ids):
    desc_idx = np.empty_like(desc_ids)
    ids2idx = numba.typed.Dict.empty(numba.int64, numba.int64)
    for i in range(len(ids)):
        ids2idx[ids[i]] = i

    progenitor_sizes = np.zeros_like(ids, dtype=np.uint32)

    for i in range(len(desc_ids)):
        if desc_ids[i] >= 0:
            dix = ids2idx[desc_ids[i]]
            desc_idx[i] = dix
            progenitor_sizes[dix] += 1
        else:
            desc_idx[i] = -1

    progenitor_offsets = np.empty_like(progenitor_sizes)

    progenitor_array_size = 0
    for i in range(len(ids)):
        progenitor_offsets[i] = progenitor_array_size
        progenitor_array_size += progenitor_sizes[i]

    progenitor_internal_offsets = np.zeros_like(progenitor_offsets)
    progenitor_array = np.empty(progenitor_array_size, dtype=np.uint32)

    for i in range(len(desc_ids)):
        if desc_ids[i] >= 0:
            dix = ids2idx[desc_ids[i]]
            progenitor_array[
                progenitor_offsets[dix] + progenitor_internal_offsets[dix]
            ] = i
            progenitor_internal_offsets[dix] += 1

    return desc_idx, progenitor_array, progenitor_offsets


@numba.jit(nopython=True)
def _fill_branch_pos_desc(
    branch_position,
    parent_branch_position,
    branch_size,
    progenitor_array,
    progenitor_offsets,
):
    for j in range(len(parent_branch_position)):
        current_pos = parent_branch_position[j] + 1
        low = progenitor_offsets[j]
        up = (
            progenitor_offsets[j + 1]
            if j + 1 < len(progenitor_offsets)
            else len(progenitor_array)
        )
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
            assert branch_position[j] != -1
    return current_pos


@numba.jit(nopython=True)
def _calculate_branch_size(branch_size, prev_branch_size, desc_index):
    for j in range(len(desc_index)):
        if desc_index[j] != -1:
            branch_size[desc_index[j]] += prev_branch_size[j]


def _fill_branch_pos(
    branch_position,
    parent_branch_position,
    branch_size,
    desc_index,
    progenitor_array,
    progenitor_offsets,
    new_branch_insert_pos,
):
    # loop over all progenitors
    if len(parent_branch_position) > 0:
        _fill_branch_pos_desc(
            branch_position,
            parent_branch_position,
            branch_size,
            progenitor_array,
            progenitor_offsets,
        )

    # loop over all with desc == -1:
    current_pos = _fill_branch_pos_new(
        branch_position, branch_size, new_branch_insert_pos, desc_index
    )
    return current_pos


def _read_data(
    partition: Partition,
    simulation: Simulation,
    filename: str,
    logger: Callable,
    fields_config: FieldsConfig,
    rebalance: bool,
    verbose: Union[bool, int],
):
    fields_xyz = [
        fields_config.node_position_x,
        fields_config.node_position_y,
        fields_config.node_position_z,
    ]

    with Timer(name="read treenodes: read genericio", logger=logger):
        data = pygio.read_genericio(
            filename,
            fields_config.read_fields,
            method=pygio.PyGenericIO.FileIO.FileIOMPI,
            rebalance_sourceranks=rebalance,
        )

    # mask invalid data
    with Timer(name="read treenodes: validate data", logger=logger):
        mask = np.ones_like(data[fields_xyz[0]], dtype=np.bool_)
        for k in data.keys():
            mask &= np.isfinite(data[k])
        n_invalid = len(mask) - np.sum(mask)
        if n_invalid > 0:
            if verbose > 1:
                print(
                    f"WARNING rank {partition.rank}: {n_invalid} invalid halos",
                    flush=True,
                )
            for k in data.keys():
                data[k] = data[k][mask]
        n_invalid_global = partition.comm.reduce(n_invalid, root=0, op=MPI.SUM)
        if partition.rank == 0 and n_invalid_global > 0:
            print(f"WARNING: {n_invalid_global} invalid halos (nan/inf)", flush=True)

    # normalize data
    for k in fields_xyz:
        data[k] = np.remainder(data[k], simulation.rl)

    # assign to rank by position
    with Timer(name="read treenodes: distribute", logger=logger):
        data = distribute(
            partition,
            simulation.rl,
            data,
            fields_xyz,
            verbose=verbose,
            verify_count=True,
        )

    # calculate derived fields
    with Timer(name="read treenodes: calculate derived fields", logger=logger):
        for k, f in fields_config.derived_fields.items():
            data[k] = f(data)

    # only keep necessary fields
    data_new = {}
    for k in fields_config.keep_fields:
        data_new[k] = data[k]

    return data_new


def _catalog2tree_step(
    step: int,
    data_store: Union[GenericIOStore, Mapping],
    index_store: Union[NumpyStore, Mapping],
    size_store: Mapping,
    local_desc: np.ndarray,
    partition: Partition,
    simulation: Simulation,
    treenode_base: str,
    fields_config: FieldsConfig,
    do_all2all_exchange: bool,
    fail_on_desc_not_found: bool,
    rebalance_gio_read: bool,
    verbose: Union[bool, int],
    logger: Callable[[str], None],
):
    logger(f"\nSTEP {step:3d}\n--------\n")

    # read data and calculate derived fields (contains its own timers)
    with Timer(name="read treenodes", logger=logger):
        if "#" in treenode_base:
            treenode_filename = treenode_base.replace("#", str(step))
        else:
            treenode_filename = f"{treenode_base}{step}.treenodes"
        data = _read_data(
            partition=partition,
            simulation=simulation,
            filename=treenode_filename,
            logger=logger,
            fields_config=fields_config,
            rebalance=rebalance_gio_read,
            verbose=verbose,
        )

    # exchange progenitors that belong to neighboring ranks
    with Timer(name="exchange descendants", logger=logger):
        if fail_on_desc_not_found:
            na_desc_key = None
        else:
            na_desc_key = -1
        data = exchange(
            partition,
            data,
            fields_config.desc_node_index,
            local_desc,
            filter_key=lambda idx: idx >= 0,
            verbose=verbose,
            do_all2all=do_all2all_exchange,
            replace_notfound_key=na_desc_key,
        )

    # sort array by descendant index, most massive first
    with Timer(name="sort arrays", logger=logger):
        s = np.lexsort(
            (data[fields_config.desc_node_index], data[fields_config.tree_node_mass])
        )[::-1]
        for k in data.keys():
            data[k] = data[k][s]

    # get index to descendant for each progenitor, and an inverse list of progenitors for each descendant
    with Timer(name="assign progenitors", logger=logger):
        desc_idx, progenitor_array, progenitor_offsets = _descendant_index(
            local_desc, data[fields_config.desc_node_index]
        )
        if len(progenitor_array) and progenitor_array.min() < 0:
            print(
                f"invalid progenitor array on rank {partition.rank}",
                file=sys.stderr,
                flush=True,
            )
            partition.comm.Abort()
    # prepare next step:
    local_ids = data["tree_node_index"]

    # store data
    data_store[step] = data
    index_store[step] = {
        "desc_idx": desc_idx,
        "progenitor_array": progenitor_array,
        "progenitor_offsets": progenitor_offsets,
    }
    size_store[step] = len(local_ids)

    return local_ids


def catalog2tree(
    simulation: Simulation,
    treenode_base: str,
    fields_config: FieldsConfig,
    output_file: str,
    *,  # The following arguments are keyword-only
    temporary_path: str = None,
    do_all2all_exchange: bool = False,
    split_output: bool = False,
    fail_on_desc_not_found: bool = True,
    rebalance_gio_read: bool = False,
    mpi_waittime: float = 0,
    logger: Callable[[str], None] = None,
    verbose: Union[bool, int] = False,
) -> None:
    """The main routine that converts treenode-catalogs to HDF5 treenode forests

    [add basic outline of algorithm]

    Parameters
    ----------
    simulation
        a :class:`Simulation` instance containing the cosmotools steps

    treenode_base
        the base path for the treenode files.
        - if ``treenode_base`` contains ``#``, ``#`` will be replace by the current step number
        - otherwise, the path will be completed by appending ``[step].treenodes``.

    fields_config
        a :class:`FieldsConfig` instance, containing the treenodes filed names

    output_file
        base path for the output file(s)

    temporary_path
        base path for temporary files. Note: folders must exist.

    do_all2all_exchange
        if ``False``, will exchange among neighboring ranks first and then
        all2all. If ``True``, will do all2all directly

    split_output
        if ``True``, forests will be stored in multiple HDF5 files (one per
        rank). If ``False``, all data will be combined in a single file (might
        not be feasible for large simulations)

    fail_on_desc_not_found
        if ``True``, will abort if a descendant halo cannot be found among all
        ranks. If ``False``, the orphaned halo will become the root of the
        subtree.

    rebalance_gio_read
        if ``True``, will reassign the reading ranks for the treenode files. Can
        be slower or faster.

    mpi_waittime
        time in seconds for which the code will wait for the MPI to be
        initialized. Can help with some MPI errors (on cooley)

    logger
        a logging function, e.g. ``print``

    verbose
        verbosity level, either 0, 1, or 2
    """
    # Set a timer for the full run
    total_timer = Timer("total time", logger=None)
    total_timer.start()

    # Initialize MPI partition
    time.sleep(mpi_waittime)
    partition = Partition(create_neighbor_topo=not do_all2all_exchange)

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
            print("  Decomposition: ", partition.decomposition)
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
                    n26 = partition.neighbor_ranks
                    print("  26-neighbors N : ", len(n26))
                    print("  26-neighbors   : ", n26)
                print("", flush=True)
            partition.comm.Barrier()

    # cosmotools steps
    steps = simulation.cosmotools_steps

    # read final snapshot (tree roots)
    with Timer(name="read treenodes", logger=logger):
        if "#" in treenode_base:
            treenode_filename = treenode_base.replace("#", str(steps[-1]))
        else:
            treenode_filename = f"{treenode_base}{steps[-1]}.treenodes"
        data = _read_data(
            partition,
            simulation,
            treenode_filename,
            logger,
            fields_config,
            rebalance=rebalance_gio_read,
            verbose=verbose,
        )

    # sort by tree_node_index
    with Timer(name="sort arrays", logger=logger):
        s = np.argsort(data[fields_config.tree_node_index])
        for k in data.keys():
            data[k] = data[k][s]

    local_ids = data[fields_config.tree_node_index]
    dtypes = {k: i.dtype for k, i in data.items()}

    # Containers to store local data from each snapshot
    data_by_step = GenericIOStore(partition, simulation.rl, temporary_path)
    data_by_step[steps[-1]] = data

    index_by_step = NumpyStore(partition, temporary_path)
    index_by_step[steps[-1]] = {
        "desc_idx": -1.0 * np.ones(len(s), dtype=np.int64),
        "progenitor_array": np.empty(0, dtype=np.uint32),
        "progenitor_offsets": np.empty(0, dtype=np.uint32),
    }
    size_by_step = {steps[-1]: len(local_ids)}

    # iterate over previous snapshots: read, assign, prepare ordering
    for step in steps[-2::-1]:
        local_ids = _catalog2tree_step(
            step,
            data_by_step,
            index_by_step,
            size_by_step,
            local_ids,
            partition,
            simulation,
            treenode_base,
            fields_config,
            do_all2all_exchange,
            fail_on_desc_not_found,
            rebalance_gio_read,
            verbose,
            logger,
        )
        gc.collect()  # explicitly free memory
        if verbose:
            debug_memory(partition, "AFTER GC")

    # local and total size of forest
    local_size = sum([l for s, l in size_by_step.items()])

    logger("\nBUILDING TREE\n-------------\n")
    # size of branches, iteratively
    with Timer("calculate branch sizes", logger=logger):
        branch_sizes = {
            step: np.ones(l, dtype=np.uint64) for step, l in size_by_step.items()
        }
        desc_index = np.empty(0, dtype=np.int64)
        prev_branch_size = np.empty(0, dtype=np.uint64)
        for i, step in enumerate(steps):
            branch_size = branch_sizes[step]
            _calculate_branch_size(branch_size, prev_branch_size, desc_index)

            prev_branch_size = branch_size
            next_indices = index_by_step[step]
            desc_index = next_indices["desc_idx"]

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
            desc_index = next_indices["desc_idx"]
            # convert array and offsets to list of lists
            # progenitor_idx = np.array_split(next_indices['progenitor_array'], next_indices['progenitor_offsets'][1:])

            branch_position = -np.ones(len(branch_size), dtype=np.int64)
            current_pos = _fill_branch_pos(
                branch_position,
                parent_branch_position,
                branch_size,
                desc_index,
                next_indices["progenitor_array"],
                next_indices["progenitor_offsets"],
                new_branch_insert_pos,
            )
            # mark end-of-table
            new_branch_insert_pos = current_pos

            # update for next iteration
            parent_branch_position = branch_position
            branch_positions[step] = branch_position

    if verbose:
        debug_memory(partition, "AFTER ARRAY POS")

    # Assert that we got all halos
    if new_branch_insert_pos != local_size:
        print(
            f"Error (rank {partition.rank}): forest size is {new_branch_insert_pos} "
            f"instead of {local_size}",
            flush=True,
        )
        partition.comm.Abort()

    # write data
    with Timer("output HDF5 forest", logger=logger):
        if split_output:
            write2multiple(
                partition,
                steps,
                local_size,
                branch_sizes,
                branch_positions,
                data_by_step,
                fields_config,
                dtypes,
                output_file,
                logger,
            )
        else:
            write2single(
                partition,
                steps,
                local_size,
                branch_sizes,
                branch_positions,
                data_by_step,
                fields_config,
                dtypes,
                output_file,
                logger,
            )

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
