# Two steady states with an exact linear solution, computed on a coarse grid.
# Run with:  python -m pytest tests   (or simply: python tests/test_steady_state.py)
import io
import contextlib
import numpy as np
from pyHeat2D.domain import Domain
from pyHeat2D.boundaryConditions import BoundaryConditions
from pyHeat2D.systemMatrix import SystemMatrix

Lx, Ly, nx, ny = 2.0, 1.0, 21, 11
k = 3.0


def steady_state(west, g_west, T_east):
    """west: 'dirichlet' or 'neumann'; east: Dirichlet T_east; south and north: insulated"""
    ncx, ncy = nx - 1, ny - 1
    dx, dy = Lx / ncx, Ly / ncy
    domain = Domain(Lx, Ly)
    centroid_pos, vertex_pos, _, _ = domain.create_mesh(nx, ny)
    Xc, Yc = centroid_pos
    dirichlet_west = west == "dirichlet"
    domain.set_boundary_conditions("west", lambda x, y: dirichlet_west, lambda x, y: not dirichlet_west, vertex_pos)
    domain.set_boundary_conditions("east", lambda x, y: True, lambda x, y: False, vertex_pos)
    domain.set_boundary_conditions("south", lambda x, y: False, lambda x, y: True, vertex_pos)
    domain.set_boundary_conditions("north", lambda x, y: False, lambda x, y: True, vertex_pos)
    boundary = BoundaryConditions(nx, ny, domain)
    boundary.customize_boundary_conditions_from_vertices(
        domain, vertex_pos,
        g_func=lambda x, y: g_west if np.isclose(x, -Lx / 2) else 0.0,  # only on the west side
        tbc_func=lambda x, y: 1.0 if np.isclose(x, -Lx / 2) else T_east)
    sm = SystemMatrix()
    aW, aE, aS, aN = sm.interior_contribution_matrix(ncx, ncy)
    T = np.zeros((ncx, ncy))
    for it in range(2):                       # very large dt: steady state
        M, rhs = sm.build_matrix(ncx, ncy, dx, dy, 1e12, 0.0, Xc, Yc, 1.0, 1.0, k,
                                 aW, boundary.bW, boundary.cW, aE, boundary.bE, boundary.cE,
                                 aS, boundary.bS, boundary.cS, aN, boundary.bN, boundary.cN,
                                 T, boundary.TBC, boundary.g, None)
        T = sm.solve_system(M, rhs).reshape(ncx, ncy)
    return Xc, T


def test_dirichlet_dirichlet():
    # T = 1 on the west side, T = 0 on the east side: T = (Lx/2 - x) / Lx
    Xc, T = steady_state("dirichlet", 0.0, 0.0)
    assert np.abs(T - (Lx / 2 - Xc) / Lx).max() < 1e-10


def test_neumann_dirichlet():
    # heat flux q entering through the west side (g > 0 along +x), T = 0 on the east side: T = q (Lx/2 - x) / k
    q = 0.6
    Xc, T = steady_state("neumann", q, 0.0)
    assert np.abs(T - q * (Lx / 2 - Xc) / k).max() < 1e-10


if __name__ == "__main__":
    with contextlib.redirect_stdout(io.StringIO()):
        test_dirichlet_dirichlet()
        test_neumann_dirichlet()
    print("all tests passed")
