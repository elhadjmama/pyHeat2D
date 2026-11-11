# pyHeat2D: 2D heat diffusion equation

Our python package is designed for the numerical simulation of the two-dimensional diffusion equation

```
rho * Cp * dT/dt = div( k * grad(T) ) + S
```

The package provides a framework for solving 2D partial differential equations (PDEs) using finite difference
methods on a staggered grid. Key features include domain discretization, boundary condition management, matrix
assembly, and solver integration.

Its main feature is **generic boundary conditions**. Each side of the domain can carry Dirichlet conditions
(imposed temperature), Neumann conditions (imposed heat flux) or a mix of the two, and the values can vary along
the side. The user only describes the conditions. The matrix of the system is generated automatically, so the
user never writes it by hand.

- Rectangular domain, cells of size `dx` x `dy`, temperature at the cell centres and heat fluxes on the faces.
- Dirichlet, Neumann and mixed boundary conditions, with values that vary in space and in time.
- Heterogeneous media: `rho`, `Cp` and `k` can be numbers or arrays with one value per cell. The conductivity
  on a face is the harmonic mean of the two cells, so the heat flux is conservative across interfaces.
- Source term `S(T, x, y, t)`.
- Implicit (backward Euler) time scheme, stable for any time step. One time step with a very large `dt` gives
  the steady state.
- Sparse direct solver (SciPy). The matrix is factorized once before the time loop.

## Installation

```bash
# from the folder of the package:
pip install .
# once the package is published on PyPI:
pip install pyHeat2D
```

Dependencies: NumPy, SciPy, SymPy and Matplotlib.

## Quick start

A Gaussian temperature in a square of side 1, with T = 0 on the four edges:

```python
import numpy as np
import matplotlib.pyplot as plt
from pyHeat2D.domain import Domain
from pyHeat2D.boundaryConditions import BoundaryConditions
from pyHeat2D.systemMatrix import SystemMatrix

# 1. domain and mesh: nx x ny vertices, (nx - 1) x (ny - 1) cells, centred on (0, 0)
Lx, Ly, nx, ny = 1.0, 1.0, 101, 101
ncx, ncy = nx - 1, ny - 1
dx, dy = Lx / ncx, Ly / ncy
domain = Domain(Lx, Ly)
centroid_pos, vertex_pos, centroid_num, contour_indices = domain.create_mesh(nx, ny)
Xc, Yc = centroid_pos

# 2. boundary conditions: type of each side, then values
for side in ["west", "east", "south", "north"]:
    domain.set_boundary_conditions(side, lambda x, y: True, lambda x, y: False, vertex_pos)  # Dirichlet
boundary = BoundaryConditions(nx, ny, domain)
boundary.customize_boundary_conditions_from_vertices(domain, vertex_pos,
                                                     g_func=lambda x, y: 0.0, tbc_func=lambda x, y: 0.0)

# 3. physical parameters and initial temperature
rho, Cp, k = 1.0, 1.0, 1.0
T = np.exp(-(Xc**2 + Yc**2) / (2 * 0.05**2))

# 4. time loop: the matrix does not change, it is factorized once
sm = SystemMatrix()
aW, aE, aS, aN = sm.interior_contribution_matrix(ncx, ncy)
dt, nt, lu = 1e-4, 50, None
for it in range(1, nt + 1):
    M, rhs = sm.build_matrix(ncx, ncy, dx, dy, dt, it * dt, Xc, Yc, rho, Cp, k,
                             aW, boundary.bW, boundary.cW, aE, boundary.bE, boundary.cE,
                             aS, boundary.bS, boundary.cS, aN, boundary.bN, boundary.cN,
                             T, boundary.TBC, boundary.g, None)
    if lu is None:
        lu = sm.factorize(M)
    T = lu.solve(rhs).reshape(ncx, ncy)

plt.contourf(Xc, Yc, T, 50, cmap="jet"); plt.colorbar(); plt.axis("equal"); plt.show()
```

## Boundary conditions

### 1. Type of each part of the edges

`set_boundary_conditions(side, dirichlet_mask, neumann_mask, vertex_pos)` takes two functions of `(x, y)`.
The first returns `True` where the condition is Dirichlet, the second where it is Neumann. They are called only
for the vertices of the given side. For example, on the west side, Dirichlet above y = 0 and Neumann below:

```python
domain.set_boundary_conditions("west", lambda x, y: y > 0, lambda x, y: y <= 0, vertex_pos)
```

Three rules:
- the four sides must be set **before** creating `BoundaryConditions(nx, ny, domain)`;
- every vertex of a side must receive **exactly one** type (never both masks `True`, never both `False`);
- a boundary face takes the type of its **first vertex**: the lower one on the west and east sides, the left
  one on the south and north sides.

`domain.visualize_boundary_conditions(vertex_pos)` shows the result.

### 2. Values

`customize_boundary_conditions_from_vertices(domain, vertex_pos, g_func, tbc_func)` evaluates `tbc_func(x, y)`
(temperature) and `g_func(x, y)` (heat flux) at the middle of each boundary face. The values are stored in
`boundary.TBC` and `boundary.g`, which are dictionaries with one array `(ncx, ncy)` per side. A corner cell
receives the value of each of its two sides.

### Sign convention of the Neumann condition

`g` is a **heat flux** (W/m2 in SI units), counted positive along **+x** on the west and east sides and along
**+y** on the south and north sides (Fourier's law: q = -k dT/dn):

| side | g > 0 means |
|---|---|
| west, south | heat **enters** the domain |
| east, north | heat **leaves** the domain |

An insulated side is `g = 0`. A basal heat flow of 60 mW/m2 entering from below is `g = 0.06` on the south side.

## Heterogeneous media and shapes

`rho`, `Cp` and `k` can be arrays `(ncx, ncy)`. A body of any shape is described by a filter on the cell centres:

```python
inside = (Xc / a)**2 + (Yc / b)**2 < 1        # ellipse of semi-axes a and b
k = np.where(inside, k_body, k_rock)

from matplotlib.path import Path              # body given by a list of points (x, y)
inside = Path(points).contains_points(np.c_[Xc.ravel(), Yc.ravel()]).reshape(Xc.shape)
```

A domain that is not a rectangle (a disk, for example, with conditions on the circle) is handled by building the
matrices `a`, `b`, `c` and the values `TBC`, `g` cell by cell: see the technical report.

## Good practice

- **Factorize again** (`lu = None`) only if `dt`, `rho`, `Cp`, `k` or the *types* of the boundary conditions
  change. The *values* `TBC` and `g`, and the source term, can change at each step without a new factorization.
- **Steady state:** use a very large time step (for example `dt = 1e12`) and two steps.
- **Source term:** a function `S(T, x, y, t)` written with SymPy functions (`sympy.exp`, not `numpy.exp`).
  The arguments are given in this order: temperature of the previous step, x, y, time. Only the order counts,
  not the names. Examples: `lambda T, x, y, t: x * y`, `lambda T, x, y, t: 0.1 * T`,
  `lambda T, x, y, t: x * y * sympy.exp(-t)`. A constant source is written `lambda T, x, y, t: S0 + 0 * x`.
  Define the function once, before the time loop: the symbolic coefficients are computed again each time a
  new function object is given.
- **Check the result:** the heat balance between the edges and the change of heat content, and a computation
  with cells twice smaller.

## Examples and tests

```
pyHeat2D/        the package: domain.py, boundaryConditions.py, systemMatrix.py
examples/        quickstart.py (Gaussian, compared with the exact solution), composite_wall.py
tests/           steady states with an exact linear solution
```

```bash
python examples/quickstart.py
python -m pytest tests        # or: python tests/test_steady_state.py
```

## Documentation

- Guene El Hadji Mama, Ndao. Makha , Ba. Oumar, *A generic two-dimensional solver for heat diffusion problems and its
  application to Earth heat transfer* (in submission): method and validation.

## Authors and licence

El Hadji Mama GUENE (Cayor Logic System), Makha NDAO (Universite Amadou Mahtar Mbow) and Oumar BA (Cayor Logic System). MIT licence, see `LICENSE`.
