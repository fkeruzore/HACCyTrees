from ..simulations import Simulation
import pygio
from mpipartition import Partition, distribute, exchange
import numpy as np
from mpi4py import MPI
from typing import Union

from ..utils.datastores import NumpyStore
import h5py

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
        "infall_fof_halo_mean_vx",
        "infall_fof_halo_mean_vy",
        "infall_fof_halo_mean_vz",
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

    # order by fof tag
    # TODO: make sure central comes first
    s = np.argsort(coreprops["fof_halo_tag"])
    coreprops = {k: d[s] for k, d in coreprops.items()}

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

        # g_idx = f.create_group("index")
        # TODO: add index data
