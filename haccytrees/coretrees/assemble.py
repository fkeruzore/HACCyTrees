from typing import Union

import h5py
import numba
import numpy as np
import pygio
from mpi4py import MPI
from mpipartition import Partition, distribute, exchange

from ..simulations import Simulation
from ..utils.datastores import NumpyStore

tracked_properties = (
    [
        "fof_halo_tag",
        "core_tag",
        "tree_node_index",
        "radius",
        "central",
        "merged",
        "vel_disp",
        "host_core",
        "infall_tree_node_mass",
        "infall_fof_halo_mass",
        "infall_fof_halo_center_x",
        "infall_fof_halo_center_y",
        "infall_fof_halo_center_z",
        "infall_fof_halo_com_vx",
        "infall_fof_halo_com_vy",
        "infall_fof_halo_com_vz",
        "infall_step",
        "infall_fof_halo_tag",
        "infall_tree_node_index",
    ]
    + [x for x in "xyz"]
    + [f"v{x}" for x in "xyz"]
)


def _read_data(partition, simulation, coreprops_filename, verbose=True):
    coreprops = pygio.read_genericio(
        coreprops_filename,
        tracked_properties,
        print_stats=verbose > 0,
    )
    # assert np.all(coreprops["fof_halo_tag"] >= 0)
    for x in "xyz":
        coreprops[x] = np.fmod(coreprops[x] + simulation.rl, simulation.rl)
    coreprops = distribute(partition, simulation.rl, coreprops, "xyz", verbose=verbose)
    return coreprops


@numba.jit(nopython=True, parallel=True)
def _subsort_centrals(coretag, central_start, central_size, host_idx, group_count):
    ncores = len(coretag)
    ncentrals = len(central_start)
    child_count = np.zeros(ncores, dtype=np.int32)
    my_direct_offset = np.zeros(ncores, dtype=np.int32)
    my_total_offset = np.zeros(ncores, dtype=np.int32)

    for i in numba.prange(ncentrals):
        _start = central_start[i]
        _size = central_size[i]
        _end = _start + _size
        # count size of each group
        for j in range(_start, _end):
            # self
            group_count[j] += 1
            # add to host recursively
            _idx = j
            while _idx != _start:
                _host = host_idx[_idx]
                group_count[_host] += 1
                _idx = _host

        # calculate offset of my group among other children of my host
        for j in range(_start + 1, _end):
            my_direct_offset[j] = child_count[host_idx[j]]
            child_count[host_idx[j]] += group_count[j]

        # calculate actual offset
        my_total_offset[_start] = _start
        for j in range(_start + 1, _end):
            my_offset = my_direct_offset[j]
            _idx = j
            while _idx != _start:
                _host = host_idx[_idx]
                my_offset += my_direct_offset[_host] + 1
                _idx = _host
            my_total_offset[j] = my_offset + _start

    return my_total_offset


def _sort_core_roots(coretag, hosttag, foftag, central, extensive_asserts=True):
    # replace hosttag of central with self (by default: 0)
    hosttag = np.copy(hosttag)
    hosttag[central > 0] = coretag[central > 0]
    if extensive_asserts:
        assert np.all(hosttag > 0)
        assert np.all(np.isin(hosttag, coretag))
        assert len(np.unique(foftag)) == np.sum(central)

    # find host index
    s_core = np.argsort(coretag)
    coretag_sorted = coretag[s_core]
    host_idx = np.searchsorted(coretag_sorted, hosttag)
    host_idx = s_core[host_idx]
    if extensive_asserts:
        assert np.all(coretag[host_idx] == hosttag)

    # primary sort: by foftag, with central first
    s_fof = np.lexsort([-central, foftag])
    central_start = np.nonzero(central[s_fof])[0]
    central_size = np.diff(np.append(central_start, len(central)))

    # host_idx for new ordering
    host_idx = np.argsort(s_fof)[host_idx][s_fof]
    if extensive_asserts:
        assert np.all(coretag[s_fof][host_idx] == hosttag[s_fof])

    group_count = np.zeros_like(coretag, dtype=np.int32)
    s = _subsort_centrals(
        coretag[s_fof], central_start, central_size, host_idx, group_count
    )
    if extensive_asserts:
        # check we have the correct sizes for the centrals
        assert np.all(group_count[central_start] == central_size)
        # check it's a permutation
        assert np.all(s >= 0)
        assert np.all(s < len(s))
        assert len(np.unique(s)) == len(s)

    return s_fof[s], host_idx[s], group_count[s]


def reorganize_coreproperties(
    partition: Partition,
    coreproperties_base: str,
    simulation: Simulation,
    output_base: str,
    *,
    verbose: Union[bool, int] = True,
):
    steps = simulation.cosmotools_steps

    # process final step
    step = steps[-1]
    if "#" in coreproperties_base:
        coreprops_filename = coreproperties_base.replace("#", str(step))
    else:
        coreprops_filename = f"{coreproperties_base}{step}.coreproperties"
    coreprops = _read_data(
        partition,
        simulation,
        coreprops_filename,
        verbose,
    )
    tracked_dtypes = {k: d.dtype for k, d in coreprops.items()}

    central_mask = coreprops["central"] == 1
    fof_tags = np.sort(coreprops["fof_halo_tag"][central_mask])
    mask_foreign_cores = np.isin(coreprops["fof_halo_tag"], fof_tags, invert=True)

    # communicate foreign cores
    if np.any(mask_foreign_cores):
        coreprops = exchange(
            partition,
            coreprops,
            "fof_halo_tag",
            fof_tags,
            filter_key=lambda idx: idx >= 0,
            verbose=True,
            do_all2all=False,
            replace_notfound_key=None,  # abort if no host found
        )

    # order by fof tag and centrals (for each fof tag, centrals come first)
    # s = np.argsort(coreprops["fof_halo_tag"])
    # s = np.lexsort([-coreprops["central"], coreprops["fof_halo_tag"]])
    s, host_idx, group_count = _sort_core_roots(
        coreprops["core_tag"],
        coreprops["host_core"],
        coreprops["fof_halo_tag"],
        coreprops["central"],
    )
    coreprops = {k: d[s] for k, d in coreprops.items()}
    root_idx = np.nonzero(coreprops["central"])[0]

    owned_core_tags = coreprops["core_tag"]
    owned_core_tags_count = len(owned_core_tags)
    owned_core_tags_length = np.ones_like(owned_core_tags, dtype=np.int64)
    owned_core_tags_order_idx = np.argsort(owned_core_tags)
    owned_core_tags_sorted = owned_core_tags[owned_core_tags_order_idx]

    npstore = NumpyStore(partition, temporary_path="tmp/tmp")

    coreprops["assign_idx"] = np.arange(owned_core_tags_count)
    npstore[step] = coreprops

    # now every rank has the host cores in its domain plus all its satellite cores
    # walk steps backwards
    for snapnum in range(len(steps) - 2, -1, -1):
        step = steps[snapnum]
        if "#" in coreproperties_base:
            coreprops_filename = coreproperties_base.replace("#", str(step))
        else:
            coreprops_filename = f"{coreproperties_base}{step}.coreproperties"

        coreprops = _read_data(
            partition,
            simulation,
            coreprops_filename,
            verbose,
        )
        new_core_tags_count = len(coreprops["core_tag"])
        new_core_tags_count_global = partition.comm.reduce(
            new_core_tags_count, op=MPI.SUM, root=0
        )

        mask_foreign_cores = np.isin(
            coreprops["core_tag"], owned_core_tags, invert=True
        )
        # communicate foreign cores
        if np.any(mask_foreign_cores):
            coreprops = exchange(
                partition,
                coreprops,
                "core_tag",
                owned_core_tags,
                verbose=False,
                do_all2all=False,
                replace_notfound_key=-1,
            )
            # discard orphan cores
            mask_orphan_cores = coreprops["core_tag"] == -1
            count_orphan_cores = np.sum(mask_orphan_cores)
            count_orphan_cores_global = partition.comm.reduce(
                count_orphan_cores, op=MPI.SUM, root=0
            )
            coreprops = {k: d[~mask_orphan_cores] for k, d in coreprops.items()}
            if partition.rank == 0:
                discarded_fraction = (
                    count_orphan_cores_global / new_core_tags_count_global
                )
                print(
                    f"Discarding {count_orphan_cores_global} orphan cores "
                    f"({100*discarded_fraction:.3f})%",
                    flush=True,
                )
            assert np.all(np.isin(coreprops["core_tag"], owned_core_tags_sorted))
            assign_idx = np.searchsorted(
                owned_core_tags_sorted,
                coreprops["core_tag"],
            )
            assert np.all(assign_idx >= 0)
            assert np.all(assign_idx < owned_core_tags_count)
            # assert np.all(owned_core_tags_sorted[assign_idx] == coreprops2["core_tag"])
            assign_idx = owned_core_tags_order_idx[assign_idx]
            assert np.all(owned_core_tags[assign_idx] == coreprops["core_tag"])
            owned_core_tags_length[assign_idx] += 1

            coreprops["assign_idx"] = assign_idx
            # store temporary data
            npstore[step] = coreprops

    for i in range(partition.nranks):
        if i == partition.rank:
            print(
                f"Rank {i}: {owned_core_tags_count} cores tracked, "
                f"{np.sum(owned_core_tags_length)} entries in total",
                flush=True,
            )
        partition.comm.Barrier()

    own_core_offsets = np.insert(np.cumsum(owned_core_tags_length), 0, 0)

    with h5py.File(f"{output_base}.{partition.rank}.hdf5", "w") as f:
        g_data = f.create_group("data")
        for k in tracked_properties:
            data = np.empty(np.sum(owned_core_tags_length), dtype=tracked_dtypes[k])
            mask = np.zeros_like(data, dtype=np.bool_)
            for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
                step = steps[snapnum]
                _d = npstore[step]
                _idx = own_core_offsets[_d["assign_idx"]] + i
                assert np.all(_idx < own_core_offsets[_d["assign_idx"] + 1])
                assert np.all(~mask[_idx])
                mask[_idx] = True
                data[_idx] = _d[k]
            assert np.all(mask)
            g_data.create_dataset(k, data=data)

        data = np.empty(np.sum(owned_core_tags_length), dtype=np.int16)

        for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
            step = steps[snapnum]
            _d = npstore[step]
            _idx = own_core_offsets[_d["assign_idx"]] + i
            assert np.all(_idx < own_core_offsets[_d["assign_idx"] + 1])
            data[_idx] = snapnum
        g_data.create_dataset("snapnum", data=data)

        g_idx = f.create_group("index")
        g_idx.create_dataset("root_idx", data=own_core_offsets[root_idx])

        for i, snapnum in enumerate(range(len(steps) - 1, -1, -1)):
            step = steps[snapnum]
            npstore.remove(step)
