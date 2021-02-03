# Define 3 mass-bins
mass_bins = [
    (1e12, 2e12),
    (1e13, 2e13),
    (1e14, 2e14)
]

relative_thresholds = [1/3, 1/4]

# a mask to select all z=0 halos
z0_mask = forest['snapnum'] == 100

# where we will store the merger probability for each mass bin
merger_probability = {}

for i, mlim in enumerate(mass_bins):
    # creating a target mask for halos at z=0 and in the mass-bin
    target_mask = np.copy(z0_mask)
    target_mask &= (forest['tree_node_mass'] > mlim[0]) 
    target_mask &= (forest['tree_node_mass'] < mlim[1])
    target_idx = forest['halo_index'][target_mask]
    print(len(target_idx))
    
    # Create a matrix of shape (ntargets, nsteps), where each row is the main
    # progenitor branch of a target. It contains the indices to the forest data,
    # and is -1 if the halo does not exist at that time
    mainbranch_index = haccytrees.mergertrees.get_mainbranch_indices(
        forest, simulation='LastJourney', target_index=target_idx
    )
    
    # mask of the matrix elements that are "filled"
    active_mask = mainbranch_index > 0
    
    # For all halos in the matrix, find the main progenitor index and the main merger index
    mainprog_index = haccytrees.mergertrees.get_nth_progenitor_indices(
        forest, progenitor_array, target_index=mainbranch_index[active_mask], n=1
    )
    mainmerger_index = haccytrees.mergertrees.get_nth_progenitor_indices(
        forest, progenitor_array, target_index=mainbranch_index[active_mask], n=2
    )
    
    # Calculate merger ratio at the locations where there are mergers
    # (i.e. where mainmerger_index > 0)
    mainprog_mass   = forest['tree_node_mass'][mainprog_index[mainmerger_index >=0]]
    mainmerger_mass = forest['tree_node_mass'][mainmerger_index[mainmerger_index >= 0]]
    merger_ratio = mainmerger_mass / mainprog_mass
    # Expand it to the "active" part of the matrix
    merger_ratio_active = np.zeros(len(mainprog_index))
    merger_ratio_active[mainmerger_index >= 0] = merger_ratio
    # Expand it to the matrix
    merger_ratio_matrix = np.zeros_like(mainbranch_index, dtype=np.float32)
    merger_ratio_matrix[active_mask] = merger_ratio_active
    
    # The probability for a halo to undergo a major merger at a specific snapshot
    # (along the main progenitor branch)
    total_halos_per_sn = np.sum(active_mask, axis=0)
    major_mergers_per_sn = np.array(
        [np.sum(merger_ratio_matrix > threshold, axis=0) 
            for threshold in relative_thresholds])
    
    merger_probability[i] = major_mergers_per_sn / total_halos_per_sn
    merger_probability[i][:, total_halos_per_sn == 0] = 0
    
    
# Get the scale factors associated with each step (matrix row)
simulation = haccytrees.Simulation.simulations['LastJourney']
scale_factors = simulation.step2a(np.array(simulation.cosmotools_steps))
# Get the time difference between steps in Gyr
lookback_times = simulation.step2lookback(np.array(simulation.cosmotools_steps))
dt = lookback_times[:-1] - lookback_times[1:]

# plotting the major merger probability at every timestep
fig, ax = plt.subplots()
for i, mlim in enumerate(mass_bins):
    for j, rt in enumerate(relative_thresholds):
        # Convert merger rate per snapshot to merger rate per Gyr
        merger_rate = merger_probability[i][j, 1:] / dt
        ax.plot(scale_factors[1:], merger_rate, 
                label=fr"$M_\mathrm{{FoF}}(z=0) \in [1, \; 2] "
                      fr"\times 10^{{{np.log10(mlim[0]):.0f}}} "
                      fr"\; h^{{-1}}M_\odot$, "
                      fr"$m_\mathrm{{th}}={rt:.1f}$",
                color=['tab:blue', 'tab:red', 'tab:orange'][i],
                linestyle=['-', '--'][j])
ax.set(
    yscale='log',
    xlabel='scale factor $a$', 
    ylabel=r'major mergers / halo / Gyr'
)
ax.legend()
fig.tight_layout()