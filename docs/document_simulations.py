import sys
from pathlib import Path

sys.path.insert(0, (Path(__file__).parent).absolute)

import haccytrees

def format_simulation(sim: haccytrees.Simulation):
    label = f".. _sim_{sim.name}:\n\n"
    title = f"{sim.name}\n" + "^" * len(sim.name) + "\n\n"
    cosmology = f"""
        .. rubric:: Cosmology

        | :math:`\Omega_m` = {sim.cosmo.Omega_m:.5f} 
        | :math:`\Omega_b` = {sim.cosmo.Omega_b:.5f} 
        | :math:`\Omega_\Lambda` = {sim.cosmo.Omega_L:.5f}
        | :math:`\sigma_8 = {sim.cosmo.s8}`
    """

    volume = f"""
        .. rubric:: Simulation Volume

        | :math:`L = {sim.rl} \; h^{{-1}}\mathrm{{Mpc}}`
        | NP = {sim.np}
        | :math:`m_p = {sim.particle_mass:.4e} \; h^{{-1}}M_\odot`
        | :math:`z_\mathrm{{start}} = {sim.zstart}`
        | :math:`n_\mathrm{{steps}} = {sim.nsteps}` 
    """

    # output_steps = sim.cosmotools_steps
    # output_redshift = [f"{s:5.2f}" for s in sim.step2z(output_steps)]
    # output_scalefactor = [f"{s:7.2e}" for s in sim.step2a(output_steps)]
    # output_steps = [f"{s:3d}" for s in output_steps]
    # output_table = ['Snapshot Number', 'HACC Step', 'Redshift', 'Scale Factor']
    # for i in range(len(output_steps)):

    return label + title + cosmology + volume

if __name__ == "__main__":
    simulations = haccytrees.Simulation.simulations
    simulation_names = list(simulations.keys())
    simulation_names.sort()

    output = "\n\n".join(format_simulation(simulations[s]) for s in simulation_names)

    with open("simulations.inc", "w") as f:
        f.write(output)
