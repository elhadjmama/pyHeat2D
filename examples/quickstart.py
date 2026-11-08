# Quick start: a Gaussian temperature in a square with T = 0 on the four edges
import numpy as np
import matplotlib.pyplot as plt
from pyHeat2D.domain import Domain
from pyHeat2D.boundaryConditions import BoundaryConditions
from pyHeat2D.systemMatrix import SystemMatrix

# 1. domain and mesh: nx x ny vertices, (nx - 1) x (ny - 1) cells
Lx, Ly, nx, ny = 1.0, 1.0, 101, 101
ncx, ncy = nx - 1, ny - 1
dx, dy = Lx / ncx, Ly / ncy
domain = Domain(Lx, Ly)
centroid_pos, vertex_pos, centroid_num, contour_indices = domain.create_mesh(nx, ny)
Xc, Yc = centroid_pos

# 2. boundary conditions: Dirichlet on the four sides, T = 0
for side in ["west", "east", "south", "north"]:
    domain.set_boundary_conditions(side, lambda x, y: True, lambda x, y: False, vertex_pos)
boundary = BoundaryConditions(nx, ny, domain)
boundary.customize_boundary_conditions_from_vertices(domain, vertex_pos,
                                                     g_func=lambda x, y: 0.0, tbc_func=lambda x, y: 0.0)

# 3. physical parameters and initial temperature
rho, Cp, k = 1.0, 1.0, 1.0
sigma = 0.05
T = np.exp(-(Xc**2 + Yc**2) / (2 * sigma**2))

# 4. time loop: the matrix does not change, it is factorized once
sm = SystemMatrix()
aW, aE, aS, aN = sm.interior_contribution_matrix(ncx, ncy)
dt, nt = 1e-4, 50
lu = None
for it in range(1, nt + 1):
    M, rhs = sm.build_matrix(ncx, ncy, dx, dy, dt, it * dt, Xc, Yc, rho, Cp, k,
                             aW, boundary.bW, boundary.cW, aE, boundary.bE, boundary.cE,
                             aS, boundary.bS, boundary.cS, aN, boundary.bN, boundary.cN,
                             T, boundary.TBC, boundary.g, None)
    if lu is None:
        lu = sm.factorize(M)
    T = lu.solve(rhs).reshape(ncx, ncy)

# 5. comparison with the exact solution (the edges are far from the Gaussian)
t = nt * dt
s2 = sigma**2 + 2 * k / (rho * Cp) * t
T_exact = sigma**2 / s2 * np.exp(-(Xc**2 + Yc**2) / (2 * s2))
print(f"t = {t:.4f}, max T = {T.max():.4f} (exact {T_exact.max():.4f}), max error = {np.abs(T - T_exact).max():.2e}")

plt.contourf(Xc, Yc, T, 50, cmap="jet")
plt.colorbar(label="T")
plt.axis("equal")
plt.title(f"Temperature at t = {t:.4f}")
plt.show()
