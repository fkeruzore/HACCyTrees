# Define 3 mass-bins
mass_bins = [
    (1e12, 2e12),
    (1e13, 2e13),
    (1e14, 2e14)
]

# a mask to select all z=0 halos
z0_mask = forest['snapnum'] == 100

# where we will store the mass evolution for each bin
mean_massfrac = {}

for i, mlim in enumerate(mass_bins):
    # creating a target mask for halos at z=0 and in the mass-bin
    target_mask = z0_mask & (forest['tree_node_mass'] > mlim[0]) \
                          & (forest['tree_node_mass'] < mlim[1])
    target_idx = forest['halo_index'][target_mask]
    
    # this will create a matrix of shape (ntargets, nsteps), where each column 
    # is the main progenitor branch of a target. It contains the indices to the 
    # forest data, and is -1 if the halo does not exist at that time
    mainbranch_index = haccytrees.mergertrees.get_mainbranch_indices(
        forest, simulation='LastJourney', target_index=target_idx
    )
    
    # Get the mass of the main branches
    active_mask = mainbranch_index != -1
    mainbranch_mass = np.zeros_like(mainbranch_index, dtype=np.float32)
    mainbranch_mass[active_mask] = forest['tree_node_mass'][mainbranch_index[active_mask]]
    
    # Normalize by the final mass
    mainbranch_mass /= mainbranch_mass[:, -1][:, np.newaxis]
    
    # Take the average over all halos
    mainbranch_mass = np.mean(mainbranch_mass, axis=0)
    mean_massfrac[i] = mainbranch_mass
    
    
# this is just to get the scale factors associated with each step (matrix row)
simulation = haccytrees.Simulation.simulations['LastJourney']
scale_factors = simulation.step2a(np.array(simulation.cosmotools_steps))

# plotting the average mass evolution
fig, ax = plt.subplots()
for i, mlim in enumerate(mass_bins):
    ax.plot(
        scale_factors, 
        mean_massfrac[i], 
        label=fr"$M_\mathrm{{FoF}}(z=0) \in [1, \; 2] "
              fr"\times 10^{{{np.log10(mlim[0]):.0f}}} "
              fr"\; h^{{-1}}M_\odot$")
ax.axhline(1, color='black')
ax.set(
    yscale='log', ylim=(1e-3, 1),
    xlabel='scale factor $a$', 
    ylabel=r'$\langle M_\mathrm{FoF} \; / \; M_\mathrm{FoF}(z=0) \rangle$'
)
ax.legend()
fig.tight_layout()