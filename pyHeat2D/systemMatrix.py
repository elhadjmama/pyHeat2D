import sympy as sy
import numpy as np
from scipy.sparse.linalg import spsolve, splu # solving a sparse matrix
from scipy import sparse

class SystemMatrix:
    def __init__(self):
        """
        The symbolic coefficients do not depend on the time step: they are computed and
        transformed into numerical functions (lambdify) only once, at the first call of
        build_matrix, and kept in self.functions.
        """
        self.functions = None            # numerical functions of the coefficients and of the rhs
        self.cached_source_term = None   # source term used to build self.functions
        # Declaration of symbolic variables
        #self.source_term = source_term if source_term else lambda Toc: 0

    def generate_coefficients(self, source_term=None):
        """
        Generate the generic coefficients for the system matrix.
        """
        if source_term is None:
            source_term = lambda Toc, Xc, Yc, t: 0

        # Declaration symbolic variables
        TW, TE, TN, TS, TC, Toc = sy.symbols('TW TE TN TS TC Toc')
        a = sy.symbols('a')
        dx, dy, d_dt, t = sy.symbols('dx dy d_dt t')
        Xc, Yc = sy.symbols('Xc Yc')
        TBCW, TBCE, TBCS, TBCN = sy.symbols('TBCW TBCE TBCS TBCN')  # dirichlet condition, per side
        gW, gE, gS, gN = sy.symbols('gW gE gS gN')  # neumann condition, per side: heat flux along +x / +y
        aW, aE, aS, aN, bW, bE, bS, bN, cW, cE, cS, cN = sy.symbols('aW, aE, aS, aN, bW, bE, bS, bN, cW, cE, cS, cN')
        kW, kE, kS, kN = sy.symbols('kW kE kS kN')  # face conductivities
        rho, Cp = sy.symbols('rho Cp')

        # expressing temperatures with generic boundary conditions
        TW1 = aW * TW + bW * (2 * a * TBCW - TC) + cW * (TC + gW * dx * a / kW)
        TE1 = aE * TE + bE * (2 * a * TBCE - TC) + cE * (TC - gE * dx * a / kE)
        TS1 = aS * TS + bS * (2 * a * TBCS - TC) + cS * (TC + gS * dy * a / kS)
        TN1 = aN * TN + bN * (2 * a * TBCN - TC) + cN * (TC - gN * dy * a / kN)

        qE = -kE * (TE1 - TC) / dx
        qW = -kW * (TC - TW1) / dx
        qN = -kN * (TN1 - TC) / dy
        qS = -kS * (TC - TS1) / dy
        symbolic_source_term = source_term(Toc, Xc, Yc, t)

        fT = (qE - qW) / dx + (qN - qS) / dy + rho * Cp * (TC - a * Toc) * d_dt - a * symbolic_source_term
        # compute the coefficients for the system matrix
        dofs = sy.Matrix([TE, TW, TS, TN, TC])
        coefficients = {}
        print("#############################################")
        print(" Generic computation of the 5 coefficients for the system matrix ")
        for i in range(len(dofs)):
            coeff = fT.diff(dofs[i])
            coefficients[str(dofs[i])] = coeff
            print(f'c{str(dofs[i])} = {str(coeff)}')
        # Compute the right-hand side
        rhs = -fT.diff(a)
        print("#############################################")
        print(" The generic right-hand side is ")
        print(f'{rhs} = {str(rhs)}')
        print("#################################################")
        return coefficients, rhs

    def interior_contribution_matrix(self, ncx, ncy):
        # Design the interior contribution matrix aW, aE, aS, aN
        aW = np.ones((ncx, ncy))
        aW[0, :] = 0
        aE = np.ones((ncx, ncy))
        aE[ncx - 1, :] = 0
        aS = np.ones((ncx, ncy))
        aS[:, 0] = 0
        aN = np.ones((ncx, ncy))
        aN[:, ncy - 1] = 0

        return aW, aE, aS, aN

    def face_conductivities(self, k1, ncx, ncy):
        """
        Conductivity on the four faces of each cell.
        k1 is given at the cell centres (scalar or array (ncx, ncy)).
        An interior face takes the harmonic mean of its two cells: this is the exact
        conductance of two half-cells in series, so the flux leaving a cell is the flux
        entering its neighbour. A boundary face takes the conductivity of its cell.
        """
        k1 = np.ones((ncx, ncy)) * k1
        kW = k1.copy()
        kE = k1.copy()
        kS = k1.copy()
        kN = k1.copy()
        kW[1:ncx, :] = 2 * k1[1:ncx, :] * k1[0:ncx - 1, :] / (k1[1:ncx, :] + k1[0:ncx - 1, :])
        kE[0:ncx - 1, :] = kW[1:ncx, :]
        kS[:, 1:ncy] = 2 * k1[:, 1:ncy] * k1[:, 0:ncy - 1] / (k1[:, 1:ncy] + k1[:, 0:ncy - 1])
        kN[:, 0:ncy - 1] = kS[:, 1:ncy]
        return kW, kE, kS, kN

    def values_per_side(self, values):
        """
        Boundary values (TBC or g) for the west, east, south and north sides.
        values is either a dict {side: array} (from BoundaryConditions) or a single array
        used for every side (previous behaviour).
        """
        if isinstance(values, dict):
            return values["west"], values["east"], values["south"], values["north"]
        return values, values, values, values

    def lambdify_coefficients(self, source_term):
        """
        Generate the five coefficients and the right-hand side with SymPy, then transform
        them into numerical functions. This is done once and not at each time step.
        """
        if source_term is None:
            source_term = lambda Toc, Xc, Yc, t: 0

        # Declaration symbolic variables
        TW, TE, TN, TS, TC, Toc = sy.symbols('TW TE TN TS TC Toc')
        a = sy.symbols('a')
        dx, dy, dt, t = sy.symbols('dx dy dt t')
        Xc, Yc = sy.symbols('Xc Yc')
        # dirichlet and neumann condition, one value per side (corner cells have two boundary faces)
        TBCW, TBCE, TBCS, TBCN = sy.symbols('TBCW TBCE TBCS TBCN')
        gW, gE, gS, gN = sy.symbols('gW gE gS gN')
        aW, aE, aS, aN, bW, bE, bS, bN, cW, cE, cS, cN = sy.symbols('aW, aE, aS, aN, bW, bE, bS, bN, cW, cE, cS, cN')
        # conductivity on the four faces of the cell (harmonic mean of the two neighbouring cells)
        kW, kE, kS, kN = sy.symbols('kW kE kS kN')
        rho, Cp = sy.symbols('rho Cp')

        # expressing temperatures with generic boundary conditions
        TW1 = aW * TW + bW * (2 * a * TBCW - TC) + cW * (TC + gW * dx * a / kW)
        TE1 = aE * TE + bE * (2 * a * TBCE - TC) + cE * (TC - gE * dx * a / kE)
        TS1 = aS * TS + bS * (2 * a * TBCS - TC) + cS * (TC + gS * dy * a / kS)
        TN1 = aN * TN + bN * (2 * a * TBCN - TC) + cN * (TC - gN * dy * a / kN)

        qE = -kE * (TE1 - TC) / dx
        qW = -kW * (TC - TW1) / dx
        qN = -kN * (TN1 - TC) / dy
        qS = -kS * (TC - TS1) / dy
        symbolic_source_term = source_term(Toc, Xc, Yc, t)
        fT = (qE - qW) / dx + (qN - qS) / dy + rho * Cp * (TC - a * Toc)/dt - a * symbolic_source_term
        # compute the coefficients for the system matrix
        dofs = sy.Matrix([TE, TW, TS, TN, TC])
        coefficients = {}
        for i in range(len(dofs)):
            coeff = fT.diff(dofs[i])
            coefficients[str(dofs[i])] = coeff
        # Compute the right-hand side
        rhs = -fT.diff(a)

        # transform sy Matrix to numpy using lambdify
        cTW = sy.lambdify((aW, kW, dx), coefficients["TW"], "numpy")
        cTE = sy.lambdify((aE, kE, dx), coefficients["TE"], "numpy")
        cTS = sy.lambdify((aS, kS, dy), coefficients["TS"], "numpy")
        cTN = sy.lambdify((aN, kN, dy), coefficients["TN"], "numpy")
        cTC = sy.lambdify((dx, dy, dt, Xc, Yc, rho, Cp, kW, kE, kS, kN, bW, cW, bE, cE, bS, cS, bN, cN), coefficients["TC"], "numpy")
        rhs = sy.lambdify((dx, dy, dt, t, Xc, Yc, rho, Cp, kW, kE, kS, kN, bW, cW, bE, cE, bS, cS, bN, cN, Toc,
                           TBCW, TBCE, TBCS, TBCN, gW, gE, gS, gN), rhs, "numpy")
        return cTW, cTE, cTS, cTN, cTC, rhs

    def build_matrix(self, ncx, ncy, dx1, dy1, dt1, time, Xc1, Yc1, rho1, Cp1, k1, aW1, bW1, cW1, aE1, bE1, cE1, aS1, bS1, cS1, aN1, bN1, cN1, Toc1, TBC1, g1, source_term):
        """
        Assemble the system matrix and RHS vector for a 2D heat equation with mixed boundary conditions.
        Generate the generic coefficients for the system matrix.
        """
        # symbolic coefficients and lambdify: only at the first call (or if the source term changes)
        if self.functions is None or source_term is not self.cached_source_term:
            self.functions = self.lambdify_coefficients(source_term)
            self.cached_source_term = source_term
        cTW, cTE, cTS, cTN, cTC, rhs = self.functions

        # numerical values: face conductivities, and boundary values for each side
        kW1, kE1, kS1, kN1 = self.face_conductivities(k1, ncx, ncy)
        TBCW1, TBCE1, TBCS1, TBCN1 = self.values_per_side(TBC1)
        gW1, gE1, gS1, gN1 = self.values_per_side(g1)

        cTW = cTW(aW1, kW1, dx1)
        cTE = cTE(aE1, kE1, dx1)
        cTS = cTS(aS1, kS1, dy1)
        cTN = cTN(aN1, kN1, dy1)
        cTC = cTC(dx1, dy1, dt1, Xc1, Yc1, rho1, Cp1, kW1, kE1, kS1, kN1, bW1, cW1, bE1, cE1, bS1, cS1, bN1, cN1)
        rhs = rhs(dx1, dy1, dt1, time, Xc1, Yc1, rho1, Cp1, kW1, kE1, kS1, kN1, bW1, cW1, bE1, cE1, bS1, cS1, bN1, cN1, Toc1,
                  TBCW1, TBCE1, TBCS1, TBCN1, gW1, gE1, gS1, gN1)
        rhs = rhs.flatten()

        # Indices for grid neighbors
        NumTe = np.arange(ncx * ncy).reshape((ncx, ncy))
        iTe = NumTe  # Center indices
        iTeW = np.zeros((ncx, ncy))
        iTeW[1:ncx, :] = NumTe[0:ncx - 1, :]
        iTeE = np.zeros((ncx, ncy))
        iTeE[0:ncx - 1, :] = NumTe[1:ncx, :]
        iTeS = np.zeros((ncx, ncy))
        iTeS[:, 1:ncy] = NumTe[:, 0:ncy - 1]
        iTeN = np.zeros((ncx, ncy))
        iTeN[:, 0:ncy - 1] = NumTe[:, 1:ncy]
        # Assemble the sparse matrix
        I = np.concatenate((iTe.flatten(), iTe.flatten(), iTe.flatten(), iTe.flatten(), iTe.flatten()))
        J = np.concatenate((iTe.flatten(), iTeW.flatten(), iTeE.flatten(), iTeN.flatten(), iTeS.flatten()))
        J = J.astype(int)  # Ensure indices are integers
        V = np.concatenate((cTC.flatten(), cTW.flatten(), cTE.flatten(), cTN.flatten(), cTS.flatten()))
        TM = sparse.csc_matrix((V, (I, J)), (ncx * ncy, ncx * ncy))

        return TM, rhs

    def solve_system(self, TM, rhs):
        Tc = spsolve(TM, rhs)
        return Tc

    def factorize(self, TM):
        """
        LU factorization of the matrix TM. When dt and the material properties do not change,
        TM is the same at each time step: it is factorized once before the time loop, and each
        step only needs lu.solve(rhs), which is much faster than spsolve.
        """
        return splu(TM.tocsc())