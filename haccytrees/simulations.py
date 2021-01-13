import dataclasses
from typing import List, ClassVar, Dict

rhoc = 2.77536627e11
@dataclasses.dataclass(frozen=True)
class Cosmology:
    name: str
    Omega_m: float
    Omega_b: float
    Omega_L: float
    h: float
    ns: float
    s8: float

    cosmologies: ClassVar[Dict] = {}

    def __post_init__(self):
        Cosmology.cosmologies[self.name] = self

    def __repr__(self):
        return f"Cosmology({self.name})"


@dataclasses.dataclass(frozen=True)
class Simulation:
    name: str
    nsteps: int
    zstart: float
    zfin: float
    rl: float
    ng: int
    np: int
    cosmo: Cosmology
    cosmotools_steps: List[int]
    fullalive_steps: List[int]

    simulations: ClassVar[Dict] = {}

    def __post_init__(self):
        Simulation.simulations[self.name] = self

    def __repr__(self):
        return f"Simulation({self.name}: RL={self.rl} NP={self.np}, NG={self.ng})"

    def step2a(self, step):
        aini = 1/(self.zstart + 1)
        afin = 1/(self.zfin + 1)
        return aini + (afin-aini)/self.nsteps * (step + 1)

    def step2z(self, step):
        return 1/self.step2a(step) - 1

    @property
    def particle_mass(self):
        return rhoc * self.cosmo.Omega_m * (self.rl/self.np)**3


# Cosmological parameters used for LJ
LastJourneyCosmo = Cosmology(
    "LastJourneyCosmo",
    Omega_m = 0.26067,
    Omega_b = 0.02242 / 0.6766**2,
    Omega_L = 1 - 0.26067,
    h = 0.6766,
    ns = 0.9665,
    s8 = 0.8102
    )

LastJourney = Simulation(
    name="LastJourney",
    cosmo=LastJourneyCosmo,
    rl=3400,
    ng=10752,
    np=10752,
    nsteps=500,
    zstart=200.,
    zfin=0.,
    cosmotools_steps=[
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 
        442, 453, 464, 475, 487, 499],
    fullalive_steps=[
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 
        442, 453, 464, 475, 487, 499]
)
    
LastJourneySV = dataclasses.replace(
    LastJourney, 
    name="LastJourneySV", 
    rl=250, 
    ng=1024, 
    np=1024)

Farpoint = Simulation(
    name="Farpoint",
    cosmo=LastJourneyCosmo,
    rl=1000,
    ng=12288,
    np=12288,
    nsteps=500,
    zstart=200.,
    zfin=0.,
    cosmotools_steps=[
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 
        442, 453, 464, 475, 487, 499],
    fullalive_steps=[
        42, 43, 44, 45, 46, 48, 49, 50, 52, 53, 54, 56, 57, 59, 60, 
        62, 63, 65, 67, 68, 70, 72, 74, 76, 77, 79, 81, 84, 86, 88, 
        90, 92, 95, 97, 100, 102, 105, 107, 110, 113, 116, 119, 121, 
        124, 127, 131, 134, 137, 141, 144, 148, 151, 155, 159, 163, 167, 
        171, 176, 180, 184, 189, 194, 198, 203, 208, 213, 219, 224, 230, 
        235, 241, 247, 253, 259, 266, 272, 279, 286, 293, 300, 307, 315, 
        323, 331, 338, 347, 355, 365, 373, 382, 392, 401, 411, 421, 432, 
        442, 453, 464, 475, 487, 499]
)