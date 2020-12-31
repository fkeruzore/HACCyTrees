from mpi4py import MPI
import argparse
import os, sys, traceback
import configparser, json
from typing import Dict, Any

from haccytrees.simulations import Simulation
from haccytrees.mergertrees.assemble import FieldsConfig, catalog2tree, DerivedFields


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nranks = comm.Get_size()


def logger(x, **kwargs):
    kwargs.pop("flush", None)
    comm.Barrier()
    if rank == 0:
        print(x, flush=True, **kwargs)


def parse_config(config: configparser.ConfigParser) -> Dict[str, Any]:
    # Input
    simulation = config['simulation']['simulation']
    simulation = Simulation.simulations[simulation]
    
    treenode_base = config['simulation']['treenode_base']
    rebalance_gio_read = config['simulation'].getboolean('rebalance_gio_read', fallback=False)

    # Output
    output_base = config['output']['output_base']
    split_output = config['output'].getboolean('split_output', fallback=True)
    temporary_path = config['output'].get('temporary_path', fallback=None)

    # Algorithm
    fail_on_desc_not_found = config['algorithm'].getboolean('fail_on_desc_not_found', fallback=True)
    do_all2all_exchange = config['algorithm'].getboolean('do_all2all_exchange', fallback=False)
    mpi_waittime = config['algorithm'].getfloat('mpi_waittime', fallback=0)
    verbose = config['algorithm'].getint('verbose', fallback=0)

    # Fields
    fields_config = FieldsConfig(**config['columns'])
    df = DerivedFields(simulation, fields_config)
    derived_fields = json.loads(config['output'].get('derived_fields', fallback="[]"))
    derived_fields = [df.derived_fields[d] for d in derived_fields]
    output_fields = json.loads(config['output']['copy_fields'])
    output_fields = [(s, s) if isinstance(s, str) else (s[0], s[1]) for s in output_fields]

    # what we need to read, keep, and output
    fields_config.read_fields = sorted(list(set(fields_config.get_essential()) | set(o[0] for o in output_fields) | set(s for d in derived_fields for s in d.requirements)))
    fields_config.keep_fields = sorted(list(set(fields_config.get_essential()) | set(o[0] for o in output_fields) | set(d.name for d in derived_fields)))
    fields_config.output_fields = output_fields
    fields_config.derived_fields = {d.name: d.function for d in derived_fields}

    return {
        'simulation': simulation,
        'treenode_base': treenode_base,
        'rebalance_gio_read': rebalance_gio_read,
        'output_base': output_base,
        'split_output': split_output,
        'temporary_path': temporary_path,
        'fail_on_desc_not_found': fail_on_desc_not_found,
        'do_all2all_exchange': do_all2all_exchange,
        'mpi_waittime': mpi_waittime,
        'verbose': verbose,
        'fields_config': fields_config
    }



if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Invalid config path: {args.config}", file=sys.stderr, flush=True)
        comm.Abort(-102)

    config = configparser.ConfigParser()
    config.read(args.config)

    config = parse_config(config)

    try:
        catalog2tree(
            simulation             = config['simulation'], 
            treenode_base          = config['treenode_base'],
            fields_config          = config['fields_config'],
            output_file            = config['output_base'], 
            temporary_path         = config['temporary_path'], 
            do_all2all_exchange    = config['do_all2all_exchange'], 
            split_output           = config['split_output'], 
            fail_on_desc_not_found = config['fail_on_desc_not_found'],
            rebalance_gio_read     = config['rebalance_gio_read'],
            mpi_waittime           = config['mpi_waittime'],
            verbose                = config['verbose']
            )
    except Exception as e:
        print(f"Uncaught error: {e}", file=sys.stderr, flush=True)
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        comm.Abort(-101)
    