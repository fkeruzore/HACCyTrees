import h5py
import numpy as np
from ..utils.timer import Timer
import gc

def write2single(partition, steps, local_size, branch_sizes, branch_positions, data_by_step, fields_write, dtypes, output_file, logger):
    with Timer("output HDF5 forest: commiunicate global offsets", logger=logger):
        # Local and global sizes
        local_sizes = partition.comm.allgather(local_size)
        total_size = np.sum(local_sizes)
        local_offsets = np.insert(np.cumsum(local_sizes)[:-1], 0, 0)
        local_offset = local_offsets[partition.rank]

        # Local and global index sizes
        local_indices_size = {}
        local_indices_sizes = {}
        local_indices_offsets = {}
        local_indices_offset = {}
        total_indices_sizes = {}
        for step in steps:
            local_indices_size[step] = len(branch_positions[step])
            local_indices_sizes[step] = partition.comm.allgather(local_indices_size[step])
            total_indices_sizes[step] = np.sum(local_indices_sizes[step])
            local_indices_offsets[step] = np.insert(np.cumsum(local_indices_sizes[step])[:-1], 0, 0)
            local_indices_offset[step] = local_indices_offsets[step][partition.rank]
    
    # Create file
    # For now: rank-by-rank (TODO: h5py MPI)
    with Timer("output HDF5 forest: create file and pre-allocate", logger=logger):
        if partition.rank == 0:
            with h5py.File(f"{output_file}.hdf5", 'w') as f:
                g = f.create_group("forest")
                for k in fields_write:
                    g.create_dataset(k, (total_size,), dtype=dtypes[k])
                g.create_dataset('branch_size', (total_size,), dtype=np.int32)
                g.create_dataset('snapnum', (total_size,), dtype=np.int16)

                # Root index
                # TODO: potentially add all snapshots
                step = steps[-1]
                gi = f.create_group(f"index_{step}")
                gi.create_dataset('index', (total_indices_sizes[step],), dtype=np.int64)
                gi.create_dataset('mass', (total_indices_sizes[step],), dtype=np.float32)
        partition.comm.Barrier()

    with Timer("output HDF5 forest: fill data", logger=logger):
        # main fields
        for k in fields_write:
            # prepare local data
            local_data = np.empty(local_size, dtype=dtypes[k])
            for step in steps:
                branch_position = branch_positions[step]
                data = data_by_step.get_field(step, k)
                local_data[branch_position] = data[k]
            # For now: rank-by-rank (TODO: h5py MPI)
            for i in range(partition.nranks):
                if partition.rank == i:
                    with h5py.File(f"{output_file}.hdf5", 'r+') as f:
                        d = f['forest'][k]
                        d[local_offset:local_offset+local_size] = local_data
                partition.comm.Barrier()
            del local_data
            gc.collect()

        # Branch size
        local_data = np.empty(local_size, np.int32)
        for step in steps:
            branch_position = branch_positions[step]
            data = branch_sizes[step]
            local_data[branch_position] = data
        # For now: rank-by-rank (TODO: h5py MPI)
        for i in range(partition.nranks):
            if partition.rank == i:
                with h5py.File(f"{output_file}.hdf5", 'r+') as f:
                    d = f['forest']['branch_size']
                    d[local_offset:local_offset+local_size] = local_data
            partition.comm.Barrier()

        # Snap Num
        local_data = -np.ones(local_size, np.int16)
        for i, step in enumerate(steps):
            branch_position = branch_positions[step]
            local_data[branch_position] = i
        # For now: rank-by-rank (TODO: h5py MPI)
        for i in range(partition.nranks):
            if partition.rank == i:
                with h5py.File(f"{output_file}.hdf5", 'r+') as f:
                    d = f['forest']['snapnum']
                    d[local_offset:local_offset+local_size] = local_data
            partition.comm.Barrier()

    with Timer("output HDF5 forest: write index", logger=logger):
        # Indices
        # TODO: potentially add all snapshots
        step = steps[-1]
        index = branch_positions[step] + local_offset
        mass = data_by_step.get_field(step, 'tree_node_mass')['tree_node_mass']

        for i in range(partition.nranks):
            if partition.rank == i:
                with h5py.File(f"{output_file}.hdf5", 'r+') as f:
                    d = f[f'index_{step}']['index']
                    d[local_indices_offset[step]:local_indices_offset[step]+local_indices_size[step]] = index
                    d = f[f'index_{step}']['mass']
                    d[local_indices_offset[step]:local_indices_offset[step]+local_indices_size[step]] = mass
            partition.comm.Barrier()


def write2multiple(partition, steps, local_size, branch_sizes, branch_positions, data_by_step, fields_write, dtypes, output_file, logger):
    rank = partition.rank
    output_file = f"{output_file}.{rank:03d}"

    local_indices_size = {}
    for step in steps:
        local_indices_size[step] = len(branch_positions[step])
    
    # Create file
    # For now: rank-by-rank (TODO: h5py MPI)
    with Timer("output HDF5 forest: create files and pre-allocate", logger=logger):
        with h5py.File(f"{output_file}.hdf5", 'w') as f:
            g = f.create_group("forest")
            for k in fields_write:
                g.create_dataset(k, (local_size,), dtype=dtypes[k])
            g.create_dataset('branch_size', (local_size,), dtype=np.int32)
            g.create_dataset('snapnum', (local_size,), dtype=np.int16)

            # Root index
            # TODO: potentially add all snapshots
            step = steps[-1]
            gi = f.create_group(f"index_{step}")
            gi.create_dataset('index', (local_indices_size[step],), dtype=np.int64)
            gi.create_dataset('mass', (local_indices_size[step],), dtype=np.float32)
        

    with Timer("output HDF5 forest: fill data", logger=logger):
        # main fields
        for k in fields_write:
            # prepare local data
            local_data = np.empty(local_size, dtype=dtypes[k])
            for step in steps:
                branch_position = branch_positions[step]
                data = data_by_step.get_field(step, k)
                local_data[branch_position] = data[k]
            with h5py.File(f"{output_file}.hdf5", 'r+') as f:
                d = f['forest'][k]
                d[:] = local_data
            del local_data
            gc.collect()

        # Branch size
        local_data = np.empty(local_size, np.int32)
        for step in steps:
            branch_position = branch_positions[step]
            data = branch_sizes[step]
            local_data[branch_position] = data
        with h5py.File(f"{output_file}.hdf5", 'r+') as f:
            d = f['forest']['branch_size']
            d[:] = local_data

        # Snap Num
        local_data = -np.ones(local_size, np.int16)
        for i, step in enumerate(steps):
            branch_position = branch_positions[step]
            local_data[branch_position] = i
        with h5py.File(f"{output_file}.hdf5", 'r+') as f:
            d = f['forest']['snapnum']
            d[:] = local_data

    with Timer("output HDF5 forest: write index", logger=logger):
        # Indices
        # TODO: potentially add all snapshots
        step = steps[-1]
        index = branch_positions[step]
        mass = data_by_step.get_field(step, 'tree_node_mass')['tree_node_mass']

        with h5py.File(f"{output_file}.hdf5", 'r+') as f:
            d = f[f'index_{step}']['index']
            d[:] = index
            d = f[f'index_{step}']['mass']
            d[:] = mass