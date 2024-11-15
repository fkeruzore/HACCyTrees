import click
from mpipartition import Partition, S2Partition, distribute, s2_distribute
from haccytrees.coretrees.assemble import (
    CoretreesAssemblyConfig,
)
from haccytrees.mergertrees.fragments import split_fragment_tag
from haccytrees.coretrees import corematrix_reader
from pathlib import Path
import pygio
import numpy as np
import h5py
from mpi4py import MPI

from haccytrees.utils.mpi_error_handler import init_mpi_error_handler

init_mpi_error_handler()

core_fields = [
    "x",
    "y",
    "z",
    "fof_halo_tag",
    "central",
    "core_tag",
    "infall_fof_halo_center_x",
    "infall_fof_halo_center_y",
    "infall_fof_halo_center_z",
]


def read_corematrix(
    partition_cube: Partition, coreforest_base: Path, config: CoretreesAssemblyConfig
):
    if partition_cube.rank == 0:
        number_of_core_files = 0
        while Path(f"{coreforest_base}.{number_of_core_files}.hdf5").exists():
            number_of_core_files += 1

    else:
        number_of_core_files = None
    number_of_core_files = partition_cube.comm.bcast(number_of_core_files, root=0)
    assert number_of_core_files > 0

    corematrix: dict[str, np.ndarray] = None

    # Just read a tiny bit of data so that we have the right fields we can send to everyone
    if number_of_core_files < partition_cube.nranks:
        if partition_cube.rank == 0:
            corematrix = corematrix_reader(
                f"{coreforest_base}.{0}.hdf5",
                config.simulation,
                include_fields=core_fields,
                calculate_host_rows=True,
                calculate_secondary_host_row=False,
                nchunks=10000,
                chunknum=0,
            )
            for k in corematrix.keys():
                corematrix[k] = np.empty(
                    (0,) + corematrix[k].shape[1:], dtype=corematrix[k].dtype
                )
            corematrix["file_idx"] = np.empty(
                (0, corematrix["x"].shape[1]), dtype=np.uint16
            )
            corematrix["row_idx"] = np.empty(
                (0, corematrix["x"].shape[1]), dtype=np.uint32
            )
            corematrix["absolute_row_idx"] = np.tile(
                corematrix["absolute_row_idx"].reshape(-1, 1),
                (1, corematrix["x"].shape[1]),
            )
        corematrix = partition_cube.comm.bcast(corematrix, root=0)

    # Actually reading data
    if partition_cube.rank == 0:
        print(f"Reading {number_of_core_files} core files")
    partition_cube.comm.Barrier()
    for i in range(partition_cube.rank, number_of_core_files, partition_cube.nranks):
        _corematrix = corematrix_reader(
            f"{coreforest_base}.{i}.hdf5",
            config.simulation,
            include_fields=core_fields,
            calculate_host_rows=True,
            calculate_secondary_host_row=False,
            nchunks=1,
            chunknum=0,
        )
        _corematrix["file_idx"] = np.full(_corematrix["x"].shape, i, dtype=np.uint16)
        _corematrix["row_idx"] = np.tile(
            np.arange(_corematrix["x"].shape[0], dtype=np.uint32).reshape(-1, 1),
            (1, _corematrix["x"].shape[1]),
        )
        _corematrix["absolute_row_idx"] = np.tile(
            _corematrix["absolute_row_idx"].reshape(-1, 1),
            (1, _corematrix["x"].shape[1]),
        )
        if corematrix is None:
            corematrix = _corematrix
        else:
            for k in corematrix.keys():
                corematrix[k] = np.concatenate([corematrix[k], _corematrix[k]], axis=0)

    return corematrix


def read_lightcone(partition_cube: Partition, lightcone_path: Path, simulation_np: int):
    """read halo lightcone and distribute halos according to their Lagrangian position

    Parameters
    ----------
    partition_cube : Partition
        partition object
    lightcone_path : Path
        path to lightcone file (genericio)
    simulation_np : int
        number of particles per side of the simulation box

    Returns
    -------
    halo_lc : dict
        dictionary containing the lightcone halos. The halos are sorted by their (non-fragmented) fof_halo_tag
    """
    lc_fields = ["x", "y", "z", "id", "a", "replication"]
    halo_lc = pygio.read_genericio(lightcone_path, lc_fields)

    # distribute LC halos according to their Lagrangian position
    # note:: halo LC only contains one fragment per fof halo, which does not need
    # to be fragment 0
    mask_fragment = halo_lc["id"] < 0
    halo_lc["fragment_idx"] = np.zeros_like(halo_lc["id"])
    halo_lc["fof_halo_tag_clean"] = np.copy(halo_lc["id"])
    (
        halo_lc["fof_halo_tag_clean"][mask_fragment],
        halo_lc["fragment_idx"][mask_fragment],
    ) = split_fragment_tag(halo_lc["id"][mask_fragment])
    assert np.all(halo_lc["fof_halo_tag_clean"] >= 0)

    halo_lc["qz"] = (halo_lc["fof_halo_tag_clean"] % simulation_np) / simulation_np
    halo_lc["qy"] = (
        (halo_lc["fof_halo_tag_clean"] // simulation_np) % simulation_np
    ) / simulation_np
    halo_lc["qx"] = (halo_lc["fof_halo_tag_clean"] // simulation_np**2) / simulation_np

    assert np.all(halo_lc["qx"] >= 0)
    assert np.all(halo_lc["qy"] >= 0)
    assert np.all(halo_lc["qz"] >= 0)
    assert np.all(halo_lc["qx"] < 1)
    assert np.all(halo_lc["qy"] < 1)
    assert np.all(halo_lc["qz"] < 1)
    halo_lc = distribute(partition_cube, 1.0, halo_lc, ["qx", "qy", "qz"])

    # order by fof_halo_tag (without fragment index)
    s = np.argsort(halo_lc["fof_halo_tag_clean"])
    halo_lc = {k: v[s] for k, v in halo_lc.items()}

    assert np.all(halo_lc["fof_halo_tag_clean"] >= 0)
    assert np.all(halo_lc["fof_halo_tag_clean"] < (1 << 41))
    assert np.all(halo_lc["replication"] >= 0)
    assert np.all(halo_lc["replication"] < (1 << 22))
    unique_id = (halo_lc["replication"].astype(np.int64) << 41) + halo_lc["fof_halo_tag_clean"]
    num_duplicates = len(unique_id) - len(np.unique(unique_id))
    max_duplicates = 0
    num_duplicated_halos = 0
    if num_duplicates > 0:
        _uq, _idx, _cnt = np.unique(unique_id, return_index=True, return_counts=True)
        _mask = _cnt > 1
        max_duplicates = np.max(_cnt)
        num_duplicated_halos = np.sum(_mask)
        # _prtidx = np.argmax(_cnt)
        # _prtindices = np.nonzero(unique_id == _uq[_prtidx])[0]
        # print(
        #     f"DEBUG:: rank {partition_cube.rank} found {np.sum(_mask)} halos with the same fof_halo_tag/replication",
        #     f"max duplicate count {_cnt[_prtidx]}",
        #     f"fof_halo_tag_clean={halo_lc['fof_halo_tag_clean'][_prtindices]}",
        #     f"fof_halo_tag={halo_lc['id'][_prtindices]}",
        #     f"replication={halo_lc['replication'][_prtindices]}",
        #     f"pos=[{halo_lc['x'][_prtindices]}, {halo_lc['y'][_prtindices]}, {halo_lc['z'][_prtindices]}]",
        #     flush=True
        #     )
        halo_lc = {k: v[_idx] for k, v in halo_lc.items()}
        unique_id = unique_id[_idx]

        # make sure it's sorted by fof_halo_tag_clean
        s = np.argsort(halo_lc["fof_halo_tag_clean"])
        halo_lc = {k: v[s] for k, v in halo_lc.items()}
        unique_id = unique_id[s]

    max_duplicates_global = partition_cube.comm.reduce(max_duplicates, op=MPI.MAX, root=0)
    num_duplicated_halos_global = partition_cube.comm.reduce(num_duplicated_halos, op=MPI.SUM, root=0)
    if partition_cube.rank == 0:
        print(f"DEBUG:: found {num_duplicated_halos_global} duplicated halos, max duplicates {max_duplicates_global}", flush=True)


    assert len(np.unique(unique_id)) == len(halo_lc["id"])

    unique_tags, unique_reverse, unique_counts = np.unique(
        halo_lc["fof_halo_tag_clean"], return_inverse=True, return_counts=True
    )

    # print("DEBUG max replications", np.max(unique_counts))
    halo_lc["replications_count"] = unique_counts[unique_reverse]

    return halo_lc


def distribute_cores_at_step(
    partition_cube: Partition, corematrix: dict, snap_num: int, simulation_np: int
):
    # Get all cores at that step
    cores_step = {k: v[:, snap_num] for k, v in corematrix.items()}
    mask = cores_step["core_tag"] > 0

    # Make sure the host has the same fof_halo_tag as the core
    # assert np.all(
    #     cores_step["fof_halo_tag"][mask]
    #     == cores_step["fof_halo_tag"][cores_step["top_host_row"][mask]]
    # )

    cores_step = {k: v[mask] for k, v in cores_step.items()}

    # distribute cores by their Lagrangian position
    mask_fragment = cores_step["fof_halo_tag"] < 0
    cores_step["fof_halo_tag_clean"] = np.copy(cores_step["fof_halo_tag"])
    cores_step["fragment_idx"] = np.zeros_like(cores_step["fof_halo_tag"])
    (
        cores_step["fof_halo_tag_clean"][mask_fragment],
        cores_step["fragment_idx"][mask_fragment],
    ) = split_fragment_tag(cores_step["fof_halo_tag"][mask_fragment])
    assert np.all(cores_step["fof_halo_tag_clean"] >= 0)
    cores_step["qz"] = (
        cores_step["fof_halo_tag_clean"] % simulation_np
    ) / simulation_np
    cores_step["qy"] = (
        (cores_step["fof_halo_tag_clean"] // simulation_np) % simulation_np
    ) / simulation_np
    cores_step["qx"] = (
        cores_step["fof_halo_tag_clean"] // simulation_np**2
    ) / simulation_np
    assert np.all(cores_step["qx"] >= 0)
    assert np.all(cores_step["qy"] >= 0)
    assert np.all(cores_step["qz"] >= 0)
    assert np.all(cores_step["qx"] < 1)
    assert np.all(cores_step["qy"] < 1)
    assert np.all(cores_step["qz"] < 1)
    cores_step = distribute(partition_cube, 1.0, cores_step, ["qx", "qy", "qz"])

    return cores_step


@click.command()
@click.argument(
    "config_file",
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--lightcone-pattern",
    required=True,
    type=str,
)
@click.option(
    "--timestep-file",
    required=True,
    type=click.Path(
        exists=True,
        dir_okay=False,
        readable=True,
        resolve_path=False,
        path_type=Path,
    ),
)
@click.option(
    "--output-base",
    required=True,
    type=str,
)
def cli(
    config_file: Path,
    lightcone_pattern: str,
    timestep_file: Path,
    output_base: Path,
):
    partition_cube = Partition(3)
    partition_s2 = S2Partition()

    if partition_s2.rank == 0:
        with open(output_base + "-decomposition.txt", "w") as f:
            f.write(f"{'index':<10} {'theta':<20} {'phi':<20}\n\n")
            for i in range(partition_s2.nranks):
                _theta = partition_s2.all_theta_extents[i]
                _theta = f"[{_theta[0]:8.6f}, {_theta[1]:8.6f}]"
                _phi = partition_s2.all_phi_extents[i]
                _phi = f"[{_phi[0]:8.6f}, {_phi[1]:8.6f}]"
                f.write(f"{i:<10} {_theta:<20} {_phi:<20}\n")

    config = CoretreesAssemblyConfig.parse_config(config_file)
    sim_np = config.simulation.np

    with open(timestep_file) as f:
        timesteps = [int(t) for t in f.read().split()]
    # +1 because of how the lightcone is constructed
    snapnums = [
        config.simulation.cosmotools_steps.index(step) + 1 for step in timesteps
    ]

    # Read all coretrees
    coreforest_base = config_file.parent / config.output_base
    corematrix = read_corematrix(partition_cube, coreforest_base, config)

    # Forward iterate over lightcone outputs
    for step, snap_num in zip(timesteps, snapnums):
        if partition_cube.rank == 0:
            _tstep = config.simulation.cosmotools_steps[snap_num]
            print(
                f"Processing step {step} (snapnum {snap_num}, target_step {_tstep})",
                flush=True,
            )
        partition_cube.comm.Barrier()

        # Read LC shell per step
        if partition_cube.rank == 0:
            print(" - Reading lightcone", flush=True)
        lightcone_catalog = lightcone_pattern.replace("#", str(step))
        halo_lc = read_lightcone(partition_cube, lightcone_catalog, sim_np)
        partition_cube.comm.Barrier()

        if partition_cube.rank == 0:
            print(" - Distribute cores at step", flush=True)
        cores_step = distribute_cores_at_step(
            partition_cube, corematrix, snap_num, sim_np
        )

        # At this point, cores and halos on the lightcone are on the same rank
        if partition_cube.rank == 0:
            print(" - Match LC with cores", flush=True)
        # Get all the cores whos parent halo intersects with the lightcone
        mask = np.isin(cores_step["fof_halo_tag_clean"], halo_lc["fof_halo_tag_clean"])
        cores_step = {k: v[mask] for k, v in cores_step.items()}

        # Sort cores by fof_halo_tag (including fragment index)
        s = np.argsort(cores_step["fof_halo_tag"])
        cores_step = {k: v[s] for k, v in cores_step.items()}

        # Match cores to halos by fof_halo_tag (without fragment index)
        assert np.all(np.diff(halo_lc["fof_halo_tag_clean"]) >= 0)  # check if sorted
        lc_index = np.searchsorted(
            halo_lc["fof_halo_tag_clean"], cores_step["fof_halo_tag_clean"]
        )
        assert np.all(lc_index < len(halo_lc["fof_halo_tag_clean"]))
        assert np.all(lc_index >= 0)
        assert np.all(
            halo_lc["fof_halo_tag_clean"][lc_index] == cores_step["fof_halo_tag_clean"]
        )

        # Find fof_halo_tag with correct fragment tag idx inside cores
        if partition_cube.rank == 0:
            print("   - find central core with matching fragment tag", flush=True)
        cores_lc_host_tag = halo_lc["id"][lc_index]
        mask_missing = ~np.isin(cores_lc_host_tag, cores_step["fof_halo_tag"])
        num_missing = np.sum(mask_missing)
        num_total = len(mask_missing)
        if num_missing > 0:
            # message = f"DEBUG rank {partition_cube.rank}:\n"
            # message += f"  Missing {np.sum(mask_missing)} fof_halo_tag (out of {len(mask_missing)})\n"
            # message += f"  Missing fof_halo_tag: {cores_lc_host_tag[mask_missing]}\n"
            # message += f"  Missing fof_halo_tag_clean: {halo_lc['fof_halo_tag_clean'][lc_index][mask_missing]}\n"
            # message += f"  Missing fragment_idx: {halo_lc['fragment_idx'][lc_index][mask_missing]}\n"
            # message += (
            #     f"  Required by cores: {cores_step['fof_halo_tag'][mask_missing]}\n"
            # )
            # print(message)

            # Remove missing halos
            cores_step = {k: v[~mask_missing] for k, v in cores_step.items()}
            cores_lc_host_tag = cores_lc_host_tag[~mask_missing]
            lc_index = lc_index[~mask_missing]

        num_missing_global = partition_cube.comm.reduce(num_missing, root=0)
        num_total_global = partition_cube.comm.reduce(num_total, root=0)
        if partition_cube.rank == 0:
            print(f"   - missing {num_missing_global} out of {num_total_global} fof_halo_tag", flush=True)
        assert np.all(np.isin(cores_lc_host_tag, cores_step["fof_halo_tag"]))
        partition_cube.comm.Barrier()

        # Handle replications
        if partition_cube.rank == 0:
            print("   - handle replications", flush=True)
        if len(lc_index) > 0:
            _counts = halo_lc["replications_count"][lc_index]
            assert np.all(_counts > 0)
            # replicate each core by the number of replications
            s = np.repeat(np.arange(len(cores_lc_host_tag)), _counts)
            cores_step = {k: v[s] for k, v in cores_step.items()}
            lc_index = lc_index[s]
            # offset each repeated index by 1 (so it points to the next replicated halo in the lightcone)
            lc_index += np.concatenate([np.arange(c) for c in _counts])
            # make sure we didn't screw up anything
            assert np.all(
                halo_lc["fof_halo_tag_clean"][lc_index]
                == cores_step["fof_halo_tag_clean"]
            )
            mask_invalid = halo_lc["id"][lc_index] != cores_lc_host_tag[s]
            if np.any(mask_invalid):
                print(f"DEBUG:: found {np.sum(mask_invalid)} mismatching fof_halo_tag")
                print(halo_lc["id"][lc_index][mask_invalid], cores_lc_host_tag[s][mask_invalid], flush=True)
            # cores_lc_host_tag = halo_lc["id"][lc_index]
            cores_lc_host_tag = cores_lc_host_tag[s]

        # Some sanity checks:
        # - we have all cores of halos in the lightcone
        assert np.all(np.isin(cores_lc_host_tag, cores_step["fof_halo_tag"]))
        cores_lc_host_idx = np.searchsorted(
            cores_step["fof_halo_tag"],cores_lc_host_tag
        )
        # - check the fof_halo_tag matches
        assert np.all(
            cores_step["fof_halo_tag"][cores_lc_host_idx] == cores_lc_host_tag
        )
        # apparently they don't need to be centrals...
        # assert np.all(cores_step["central"][cores_lc_host_idx] > 0)

        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - calculate core offsets", flush=True)
        # Calculate distances
        mask_valid = np.ones_like(cores_step["x"], dtype=np.bool_)
        for x in "xyz":
            _dx = (
                cores_step[x]
                - cores_step[x][cores_lc_host_idx]
                # - cores_step[f"infall_fof_halo_center_{x}"][cores_lc_host_idx]
            )
            _dx[_dx > config.simulation.rl / 2] -= config.simulation.rl
            _dx[_dx < -config.simulation.rl / 2] += config.simulation.rl
            if not np.all(np.abs(_dx) <= 20):
                print("DEBUG:: found large dx", x, _dx[np.abs(_dx) > 20], flush=True)
            mask_valid &= np.abs(_dx) <= 20
            # assert np.all(np.abs(_dx) <= 20)
            cores_step[f"d{x}"] = _dx
        cores_step = {k: v[mask_valid] for k, v in cores_step.items()}
        lc_index = lc_index[mask_valid]

        for x in "xyz":
            cores_step[x] = halo_lc[x][lc_index] + cores_step[f"d{x}"]
            cores_step[f"host_{x}"] = halo_lc[x][lc_index]
        r = np.sqrt(np.sum([cores_step[x] ** 2 for x in "xyz"], axis=0))
        rhost = np.sqrt(np.sum([cores_step[f"host_{x}"] ** 2 for x in "xyz"], axis=0))

        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - calculate angular coordinates", flush=True)
        # Calculate angular lightcone coordinates
        # theta between [0, pi], phi between [0, 2pi]
        cores_step["theta"] = np.arccos(cores_step["z"] / r)
        cores_step["phi"] = np.arctan2(cores_step["y"], cores_step["x"]) + np.pi
        cores_step["host_theta"] = np.arccos(cores_step["host_z"] / rhost)
        cores_step["host_phi"] = (
            np.arctan2(cores_step["host_y"], cores_step["host_x"]) + np.pi
        )
        cores_step["scale_factor"] = halo_lc["a"][lc_index]

        assert np.all(cores_step["theta"] >= 0)
        assert np.all(cores_step["theta"] <= np.pi)
        assert np.all(cores_step["phi"] >= 0)
        assert np.all(cores_step["phi"] <= 2 * np.pi)
        assert np.all(cores_step["host_theta"] >= 0)
        assert np.all(cores_step["host_theta"] <= np.pi)
        assert np.all(cores_step["host_phi"] >= 0)
        assert np.all(cores_step["host_phi"] <= 2 * np.pi)

        # if phi is 2pi, set it to 0
        cores_step["phi"] = np.fmod(cores_step["phi"], 2 * np.pi)
        cores_step["host_phi"] = np.fmod(cores_step["host_phi"], 2 * np.pi)

        partition_cube.comm.Barrier()
        if partition_cube.rank == 0:
            print(" - distribute cores on LC", flush=True)
        # distribute cores by the angular position of the host halo
        cores_step = s2_distribute(
            partition_s2,
            cores_step,
            theta_key="host_theta",
            phi_key="host_phi",
        )

        if partition_cube.rank == 0:
            print(" - write HDF5", flush=True)
        # Write cores to file
        output_file = output_base + f"-{step}.{partition_cube.rank}.hdf5"
        output_fields = core_fields + [
            "file_idx",
            "row_idx",
            "theta",
            "phi",
            "scale_factor",
        ]
        with h5py.File(output_file, "w") as f:
            for k in output_fields:
                f.create_dataset(k, data=cores_step[k])
            f.attrs["snapnum"] = snap_num
            f.attrs["theta_extent"] = partition_s2.theta_extent
            f.attrs["phi_extent"] = partition_s2.phi_extent

        partition_s2.comm.Barrier()
