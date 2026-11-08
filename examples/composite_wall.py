# Heat transfer in a composite wall (section 6): transient cooling, then comparison of the steady state
# with the two classical one-dimensional bounds.
import numpy as np
import matplotlib.pyplot as plt
from pyHeat2D.domain import Domain
from pyHeat2D.boundaryConditions import BoundaryConditions
from pyHeat2D.systemMatrix import SystemMatrix


def boundaries_choice_and_viz(vertex_pos, domain):
    # west (interior surface) and east (exterior surface): Dirichlet
    domain.set_boundary_conditions("west", lambda x, y: True, lambda x, y: False, vertex_pos)
    domain.set_boundary_conditions("east", lambda x, y: True, lambda x, y: False, vertex_pos)
    # north and south: insulated (Neumann, g = 0)
    domain.set_boundary_conditions("north", lambda x, y: False, lambda x, y: True, vertex_pos)
    domain.set_boundary_conditions("south", lambda x, y: False, lambda x, y: True, vertex_pos)


def material(prop_roc, prop_pla, prop_bois, prop_ver):
    """property of each cell: plaster | (fiberglass on top, rock wool below) | wood"""
    x = Xc + Lx / 2                        # distance from the interior surface
    y = Yc + Ly / 2                        # height from the bottom of the wall
    custom_function = np.where(y > h_roc, prop_ver, prop_roc)
    custom_function = np.where(x < e_pla, prop_pla, custom_function)
    custom_function = np.where(x > e_pla + e_roc, prop_bois, custom_function)
    return custom_function


if __name__ == '__main__':
    # plaster
    k_pla, e_pla, Cp_pla, density_pla = 0.17, 10e-3, 1000, 800
    # rock wool (lower panel) and fiberglass (upper panel), same thickness
    k_roc, e_roc, h_roc, Cp_roc, density_roc = 0.042, 100e-3, 0.1, 800, 100
    k_ver, e_ver, h_ver, Cp_ver, density_ver = 0.039, 100e-3, 0.1, 800, 10
    # wood
    k_bois, e_bois, Cp_bois, density_bois = 0.12, 20e-3, 1200, 500
    T_int, T_ext = 20.0, -15.0             # temperatures in degrees Celsius

    # domain: 1 mm cells across the wall, 5 mm along the height
    Lx = e_pla + e_roc + e_bois
    Ly = h_roc + h_ver
    nx, ny = 131, 41
    ncx, ncy = nx - 1, ny - 1
    dx, dy = Lx / ncx, Ly / ncy
    domain = Domain(Lx, Ly)
    centroid_pos, vertex_pos, centroid_num, contour_indices = domain.create_mesh(nx, ny)
    Xc, Yc = centroid_pos
    boundaries_choice_and_viz(vertex_pos, domain)

    density = material(density_roc, density_pla, density_bois, density_ver)
    heat_capacity = material(Cp_roc, Cp_pla, Cp_bois, Cp_ver)
    conductivity = material(k_roc, k_pla, k_bois, k_ver)

    boundary = BoundaryConditions(nx, ny, domain)
    g_func = lambda x, y: 0.0
    tbc_func = lambda x, y: T_int if x < 0 else T_ext
    boundary.customize_boundary_conditions_from_vertices(domain, vertex_pos, g_func, tbc_func)

    # initial state: the whole wall at 20 C; at t = 0 the exterior drops to -15 C
    sm = SystemMatrix()
    aW, aE, aS, aN = sm.interior_contribution_matrix(ncx, ncy)
    Toc = T_int * np.ones((ncx, ncy))
    dt, nt = 300.0, 288                    # 5 minutes, 24 hours
    lu, history, snapshots = None, [], {}
    for it in range(1, nt + 1):
        M, rhs = sm.build_matrix(ncx, ncy, dx, dy, dt, it * dt, Xc, Yc, density, heat_capacity, conductivity,
                                 aW, boundary.bW, boundary.cW, aE, boundary.bE, boundary.cE,
                                 aS, boundary.bS, boundary.cS, aN, boundary.bN, boundary.cN,
                                 Toc, boundary.TBC, boundary.g, None)
        if lu is None:
            lu = sm.factorize(M)
        Toc = lu.solve(rhs).reshape(ncx, ncy)
        # heat flow through the interior and exterior surfaces, per metre of wall depth (W/m)
        Q_int = np.sum(conductivity[0, :] * 2 * (T_int - Toc[0, :]) / dx) * dy
        Q_ext = np.sum(conductivity[-1, :] * 2 * (Toc[-1, :] - T_ext) / dx) * dy
        history.append((it * dt / 3600, Q_int, Q_ext))
        if it * dt / 3600 in (0.5, 2, 24):
            snapshots[it * dt / 3600] = Toc.copy()

    # steady state: one step with a very large time step
    M, rhs = sm.build_matrix(ncx, ncy, dx, dy, 1e12, 1e12, Xc, Yc, density, heat_capacity, conductivity,
                             aW, boundary.bW, boundary.cW, aE, boundary.bE, boundary.cE,
                             aS, boundary.bS, boundary.cS, aN, boundary.bN, boundary.cN,
                             Toc, boundary.TBC, boundary.g, None)
    T_steady = sm.factorize(M).solve(rhs).reshape(ncx, ncy)
    Q_steady = np.sum(conductivity[0, :] * 2 * (T_int - T_steady[0, :]) / dx) * dy
    dT = T_int - T_ext
    R = lambda k_mid: e_pla / k_pla + e_roc / k_mid + e_bois / k_bois        # m2.K/W of one panel
    Q_parallel = dT * (h_roc / R(k_roc) + h_ver / R(k_ver))                 # adiabatic paths
    Q_series = dT / (e_pla / (k_pla * Ly) + e_roc / (k_roc * h_roc + k_ver * h_ver) + e_bois / (k_bois * Ly))
    hist = np.array(history)
    print(f"Q after 24 h: interior {hist[-1, 1]:.4f} W/m, exterior {hist[-1, 2]:.4f} W/m")
    print(f"steady state: Q = {Q_steady:.5f} W/m, 1D bounds {Q_parallel:.5f} and {Q_series:.5f} W/m")
    within = hist[np.all(np.abs(hist[:, 1:] - Q_steady) < 0.01 * Q_steady, axis=1), 0]
    print(f"both surface heat flows within 1 % of the steady value after {within[0]:.2f} h; "
          f"exterior heat flow at the first step {hist[0, 2]:.1f} W/m")
    np.savez("composite_wall_results.npz", hist=hist, T05=snapshots[0.5], T2=snapshots[2], T24=snapshots[24],
             T_steady=T_steady, Xc=Xc, Yc=Yc, k=conductivity, Q=[Q_steady, Q_parallel, Q_series])

    # figures
    ext = [0, Lx * 1e3, 0, Ly * 1e3]
    fig, axs = plt.subplots(1, 4, figsize=(15, 4.2), constrained_layout=True)
    for ax, (key, title) in zip(axs[:3], ((0.5, "t = 30 min"), (2, "t = 2 h"), (24, "t = 24 h"))):
        im = ax.imshow(snapshots[key].T, origin="lower", extent=ext, aspect="auto", cmap="jet", vmin=T_ext, vmax=T_int)
        ax.set_title(title); ax.set_xlabel("x (mm)")
        for xl in (e_pla * 1e3, (e_pla + e_roc) * 1e3):
            ax.axvline(xl, color="black", lw=0.6)
        ax.axhline(h_roc * 1e3, xmin=e_pla / Lx, xmax=(e_pla + e_roc) / Lx, color="black", lw=0.6)
    axs[0].set_ylabel("y (mm)")
    fig.colorbar(im, ax=axs[:3], label="T (°C)", shrink=0.9)
    axs[3].plot(hist[:, 0], hist[:, 1], label="interior surface")
    axs[3].plot(hist[:, 0], hist[:, 2], label="exterior surface")
    axs[3].axhline(Q_steady, color="black", ls="--", lw=0.9, label="steady state")
    axs[3].set_xlim(0, 8); axs[3].set_ylim(0, 10)
    axs[3].set_xlabel("time (h)"); axs[3].set_ylabel("heat flow (W per m of wall)"); axs[3].legend()
    axs[3].set_title("Heat flow through the surfaces")
    plt.savefig("fig_wall_results.png", dpi=150)
    plt.show()
