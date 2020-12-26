import argparse
import sys, traceback
import numpy as np
from mpi4py import MPI
from haccytrees.simulations import simulation_lut, _Simulation
from haccytrees.mergertrees.catalogs2trees import catalog2tree

fields_copy = [
    "tree_node_index",
    "desc_node_index",
    "tree_node_mass",
    "fof_halo_tag",
    "fof_halo_count",
    "fof_halo_mass",
    "sod_halo_count",
    "sod_halo_mass",
    "sod_halo_radius",
    "sod_halo_cdelta",
    "sod_halo_c_acc_mass",
    "sod_halo_c_peak_mass",
    "sod_halo_cdelta_error",
    "fof_halo_center_x",
    "fof_halo_center_y",
    "fof_halo_center_z",
]

# SOD xoff
def xoff_sod(data, simulation: _Simulation):
    rl = simulation.rl
    dx = data['sod_halo_mean_x'] - data['sod_halo_min_pot_x']
    dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
    dy = data['sod_halo_mean_y'] - data['sod_halo_min_pot_y']
    dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
    dz = data['sod_halo_mean_z'] - data['sod_halo_min_pot_z']
    dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
    dd = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.array(dd / data['sod_halo_radius'], dtype=np.float32)

# FoF xoff
def xoff_fof(data, simulation: _Simulation):
    rl = simulation.rl
    dx = data['fof_halo_com_x'] - data['fof_halo_center_x']
    dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
    dy = data['fof_halo_com_y'] - data['fof_halo_center_y']
    dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
    dz = data['fof_halo_com_z'] - data['fof_halo_center_z']
    dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
    dd = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.array(dd / data['sod_halo_radius'], dtype=np.float32)

# CoM xoff
def xoff_com(data, simulation: _Simulation):
    rl = simulation.rl
    dx = data['fof_halo_com_x'] - data['sod_halo_mean_x']
    dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
    dy = data['fof_halo_com_y'] - data['sod_halo_mean_y']
    dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
    dz = data['fof_halo_com_z'] - data['sod_halo_mean_z']
    dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
    dd = np.sqrt(dx**2 + dy**2 + dz**2)
    return np.array(dd / data['sod_halo_radius'], dtype=np.float32)


fields_derived = {
    'xoff_fof': (['fof_halo_com_x', 'fof_halo_com_y', 'fof_halo_com_z', 'fof_halo_center_x', 'fof_halo_center_y', 'fof_halo_center_z', 'sod_halo_radius'], xoff_fof),
    'xoff_sod': (['sod_halo_mean_x', 'sod_halo_mean_y', 'sod_halo_mean_z', 'sod_halo_min_pot_x', 'sod_halo_min_pot_y', 'sod_halo_min_pot_z', 'sod_halo_radius'], xoff_sod),
    'xoff_com': (['sod_halo_mean_x', 'sod_halo_mean_y', 'sod_halo_mean_z', 'fof_halo_com_x', 'fof_halo_com_y', 'fof_halo_com_z', 'sod_halo_radius'], xoff_com),
}


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()

def logger(x, **kwargs):
    kwargs.pop("flush", None)
    comm.Barrier()
    if rank == 0:
        print(x, flush=True, **kwargs)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("simulation", choices=list(simulation_lut.keys()))
    parser.add_argument("treenode_base", type=str)
    parser.add_argument("output_file")
    parser.add_argument("--temporary_path")
    parser.add_argument("--do-all2all-exchange", action='store_true')
    parser.add_argument("--split-output", action='store_true')
    parser.add_argument("--continue-on-desc-not-found", action='store_true')
    parser.add_argument("--rebalance-gio-read", action='store_true')
    parser.add_argument("--mpi-waittime", type=float, default=0)
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()
    simulation = simulation_lut[args.simulation]

    # make sure to catch errors and call MPI Abort
    try:
        catalog2tree(simulation, args.treenode_base, fields_copy, fields_derived, 
                    output_file=args.output_file, 
                    temporary_path=args.temporary_path, 
                    do_all2all_exchange=args.do_all2all_exchange, 
                    split_output=args.split_output, 
                    fail_on_desc_not_found=not args.continue_on_desc_not_found,
                    rebalance_gio_read=args.rebalance_gio_read,
                    mpi_waittime=args.mpi_waittime,
                    verbose=args.verbose)
    except Exception as e:
        print(f"Uncaught error: {e}")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        comm.Abort(-101)
