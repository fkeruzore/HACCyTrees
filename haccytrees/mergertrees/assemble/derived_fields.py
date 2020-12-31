from typing import Callable, Mapping, List, Union, Sequence
from dataclasses import dataclass
import numpy as np

from ...simulations import Simulation
from .fieldsconfig import FieldsConfig

_value_no_sod = -101

@dataclass
class DerivedField:
    name: str
    requirements: List[str]
    function: Callable 

class DerivedFields:
    def __init__(self, 
            simulation: Simulation, 
            fields_config: FieldsConfig
            ):
        self.simulation= simulation
        self.fields_config = fields_config

        # Derrived Fields, Data Requirements, Functions
        self.derived_fields = {
            'xoff_sod': DerivedField('xoff_sod', [fields_config.sod_halo_radius, fields_config.sod_halo_mass, fields_config.sod_halo_com_x, fields_config.sod_halo_com_y, fields_config.sod_halo_com_z, fields_config.sod_halo_center_x, fields_config.sod_halo_center_y, fields_config.sod_halo_center_z], self.xoff_sod),
            'xoff_fof': DerivedField('xoff_fof', [fields_config.sod_halo_radius, fields_config.sod_halo_mass, fields_config.fof_halo_com_x, fields_config.fof_halo_com_y, fields_config.fof_halo_com_z, fields_config.fof_halo_center_x, fields_config.fof_halo_center_y, fields_config.fof_halo_center_z], self.xoff_fof),
            'xoff_com': DerivedField('xoff_com', [fields_config.sod_halo_radius, fields_config.sod_halo_mass, fields_config.sod_halo_com_x, fields_config.sod_halo_com_y, fields_config.sod_halo_com_z, fields_config.fof_halo_com_x   , fields_config.fof_halo_com_y   , fields_config.fof_halo_com_z   ], self.xoff_com)
        }

    # --------------------------------------------------------------------------

    def xoff_sod(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        rl = self.simulation.rl
        mask_sod = data[self.fields_config.sod_halo_mass] > 0

        dx = data[self.fields_config.sod_halo_com_x] - data[self.fields_config.sod_halo_center_x]
        dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
        dy = data[self.fields_config.sod_halo_com_y] - data[self.fields_config.sod_halo_center_y]
        dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
        dz = data[self.fields_config.sod_halo_com_z] - data[self.fields_config.sod_halo_center_z]
        dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
        xoff = np.sqrt(dx**2 + dy**2 + dz**2) / data[self.fields_config.sod_halo_radius]

        result = np.empty_like(xoff, dtype=np.float32)
        result[:] = _value_no_sod
        result[mask_sod] = xoff[mask_sod]
        return result


    # FoF xoff
    def xoff_fof(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        rl = self.simulation.rl
        mask_sod = data[self.fields_config.sod_halo_mass] > 0

        dx = data[self.fields_config.fof_halo_com_x] - data[self.fields_config.fof_halo_center_x]
        dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
        dy = data[self.fields_config.fof_halo_com_y] - data[self.fields_config.fof_halo_center_y]
        dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
        dz = data[self.fields_config.fof_halo_com_z] - data[self.fields_config.fof_halo_center_z]
        dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
        xoff = np.sqrt(dx**2 + dy**2 + dz**2) / data[self.fields_config.sod_halo_radius]

        result = np.empty_like(xoff, dtype=np.float32)
        result[:] = _value_no_sod
        result[mask_sod] = xoff[mask_sod]
        return result


    # CoM xoff
    def xoff_com(self, data: Mapping[str, np.ndarray]) -> np.ndarray:
        rl = self.simulation.rl
        mask_sod = data[self.fields_config.sod_halo_mass] > 0

        dx = data[self.fields_config.fof_halo_com_x] - data[self.fields_config.sod_halo_com_x]
        dx += (dx < -0.5*rl)*rl - (dx > 0.5*rl)*rl
        dy = data[self.fields_config.fof_halo_com_y] - data[self.fields_config.sod_halo_com_y]
        dy += (dy < -0.5*rl)*rl - (dy > 0.5*rl)*rl
        dz = data[self.fields_config.fof_halo_com_z] - data[self.fields_config.sod_halo_com_z]
        dz += (dz < -0.5*rl)*rl - (dz > 0.5*rl)*rl
        xoff = np.sqrt(dx**2 + dy**2 + dz**2) / data[self.fields_config.sod_halo_radius]

        result = np.empty_like(xoff, dtype=np.float32)
        result[:] = _value_no_sod
        result[mask_sod] = xoff[mask_sod]
        return result
