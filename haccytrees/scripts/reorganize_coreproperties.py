from mpipartition import Partition
from haccytrees.coretrees.assemble import reorganize_coreproperties
from haccytrees.simulations import Simulation
import click


@click.command()
@click.argument("coreproperties_base", type=str)
@click.argument("simulation_name", type=str)
@click.argument("output_base", type=str)
def cli(coreproperties_base, simulation_name, output_base):
    partition = Partition(create_neighbor_topo=True)
    simulation = Simulation.simulations[simulation_name]
    reorganize_coreproperties(
        partition,
        coreproperties_base,
        simulation,
        output_base,
    )


if __name__ == "__main__":
    cli()
