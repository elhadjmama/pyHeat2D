"""
pyHeat2D: a generic two-dimensional solver for the heat diffusion equation
rho * Cp * dT/dt = div( k * grad(T) ) + S, with finite differences on a staggered grid.
"""
from pyHeat2D.domain import Domain
from pyHeat2D.boundaryConditions import BoundaryConditions
from pyHeat2D.systemMatrix import SystemMatrix

__version__ = "0.0.1"
__all__ = ["Domain", "BoundaryConditions", "SystemMatrix"]
