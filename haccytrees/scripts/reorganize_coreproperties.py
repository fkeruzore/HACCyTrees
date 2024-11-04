from mpipartition import Partition
from haccytrees.coretrees.assemble import (
    reorganize_coreproperties,
    CoretreesAssemblyConfig,
)
import click
import os
from pathlib import Path

from haccytrees.utils.mpi_error_handler import init_mpi_error_handler

init_mpi_error_handler()


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
def cli(config_file: Path):
    config = CoretreesAssemblyConfig.parse_config(config_file)
    partition = Partition(create_neighbor_topo=True)

    if partition.rank == 0 and not os.path.exists(config.temporary_path):
        os.makedirs(config.temporary_path)

    partition.comm.Barrier()
    reorganize_coreproperties(partition, config)


if __name__ == "__main__":
    cli()
