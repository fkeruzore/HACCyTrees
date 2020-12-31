from dataclasses import dataclass, field
from typing import List, Tuple, Set, Mapping, Callable

@dataclass
class FieldsConfig:
    # Needed for the code
    node_position_x: str = "fof_halo_center_x"
    node_position_y: str = "fof_halo_center_y"
    node_position_z: str = "fof_halo_center_z"

    fof_halo_center_x: str = "fof_halo_center_x"
    fof_halo_center_y: str = "fof_halo_center_y"
    fof_halo_center_z: str = "fof_halo_center_z"

    fof_halo_com_x: str = "fof_halo_mean_x"
    fof_halo_com_y: str = "fof_halo_mean_y"
    fof_halo_com_z: str = "fof_halo_mean_z"

    sod_halo_center_x: str = "fof_halo_min_pot_x"
    sod_halo_center_y: str = "fof_halo_min_pot_y"
    sod_halo_center_z: str = "fof_halo_min_pot_z"

    sod_halo_com_x: str = "sod_halo_mean_x"
    sod_halo_com_y: str = "sod_halo_mean_y"
    sod_halo_com_z: str = "sod_halo_mean_z"

    sod_halo_mass  : str = "sod_halo_mass"
    sod_halo_radius: str = "sod_halo_radius"
    tree_node_index: str = "tree_node_index"
    desc_node_index: str = "desc_node_index"
    tree_node_mass : str = "tree_node_mass"

    # What we need to read (needs to be the same order on every rank!)
    read_fields: List[str] = field(default_factory=list)
    # What we need to keep until the end (needs to be the same order on every rank!)
    keep_fields: List[str] = field(default_factory=list)
    # What we need to write to the output (with renaming)
    output_fields: List[Tuple[str, str]] = field(default_factory=list)
    # What we need to additionally compute
    derived_fields: Mapping[str, Callable] = field(default_factory=dict)

    # The fields that are always necessary
    def get_essential(self):
        return [
            self.tree_node_index, 
            self.desc_node_index, 
            self.tree_node_mass, 
            self.node_position_x, 
            self.node_position_y, 
            self.node_position_z
            ]
