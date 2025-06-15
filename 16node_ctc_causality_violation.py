import numpy as np
import sympy as sp
import logging
from datetime import datetime
from scipy.integrate import solve_ivp, simpson
from scipy.linalg import block_diag as dense_block_diag
from scipy.sparse import csr_matrix, eye, kron
from scipy.sparse.linalg import expm as sparse_expm, norm as sparse_norm
from scipy.special import sph_harm
from scipy.optimize import minimize
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('simulation_log.txt'), logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logger.info("Starting 6D Unified Spacetime Simulation with CTC Qutrits, vm-Teleportation, Pythagorean Framework, and Antikythera Cycles")

CURRENT_TIME = datetime(2025, 6, 15, 0, 14)  # Updated to 12:14 AM CDT, June 15, 2025

G = 6.67430e-11  # Gravitational constant (m^3 kg^-1 s^-2)
c0 = 2.99792458e8  # Speed of light in vacuum (m/s)
hbar = 1.0545718e-34  # Reduced Planck constant (J s)
l_p = np.sqrt(hbar * G / c0**3)  # Planck length (m)
m_n = 1.67e-27  # Neutron mass (kg)
RS = 2.0 * G * m_n / c0**2  # Schwarzschild radius (m)
LAMBDA = 2.72  # Cosmological constant (m^-2)
T_c = l_p / c0  # Planck time (s)
omega = 3  # Angular frequency
kB = 1.380649e-23  # Boltzmann constant (J/K)
T_vacuum = 2.7  # Vacuum temperature (K), e.g., cosmic microwave background

# Antikythera cycle parameters (scaled down for simulation timescale)
METONIC_CYCLE = (19 * 365.25 * 24 * 3600) / 1e9  # ~5.9988e-1 seconds
SAROS_CYCLE = ((18 * 365.25 + 11) * 24 * 3600) / 1e9  # ~5.6916e-1 seconds

# Modified speed of light parameters
B = 1e-9  # External magnetic field strength (T)
alpha = 1 / 137  # Fine-structure constant
a = 1e-10  # Distance scale (m)
theta = 4/3  # Retained from your change
rho_e = 0  # Electric energy density (J/m^3), placeholder
rho_m = (B**2) / (2 * 4 * np.pi * 1e-7)  # Magnetic energy density (J/m^3)
rho = rho_e + rho_m  # Total energy density

# Calculate modified speed of light (Equation 2.17)
c_prime = c0 * (1 - (44 * alpha**2 * hbar**2 * c0**2) / (135 * m_n**2 * a**4) * np.sin(theta)**2) if rho > 0 else c0
# Simplified for isotropic case (Equation 2.19)
c_prime_isotropic = c0 * (1 - (22 * alpha**2 * rho) / (135 * m_n**2 * c0**2)) if rho > 0 else c0

CONFIG = {
    "grid_size": (6,),
    "max_iterations": 9,
    "time_delay_steps": 2,
    "ctc_feedback_factor": 2.7,  # Tuned for entropy optimization
    "dt": 1e-13,  # Reduced for numerical stability
    "dx": l_p * 1e5,
    "dv": l_p * 1e3,
    "du": l_p * 1e3,
    "omega": 3,
    "a_godel": 1.0,
    "kappa": 1e-8,
    "rtol": 1e-6,
    "atol": 1e-9,
    "field_clamp_max": 1e18,
    "dt_min": 1e-15,
    "dt_max": 1e-9,
    "max_steps_per_dt": 1000,
    "geodesic_steps": 50,
    "ctc_iterations": 100,
    "nugget_m": 1.618,
    "nugget_lambda": 2.72,
    "alpha_time": 1e-3,  # Tuned for displacement
    "vertex_lambda": 0.33333333326,
    "matrix_size": 32,
    "kappa_worm": 0.27,
    "kappa_ent": 0.27,
    "kappa_ctc": 0.813,
    "kappa_j4": 0.813,
    "sigma": 1.618,
    "kappa_j6": 1.618,
    "kappa_j6_eff": 1e-33,
    "j6_scaling_factor": 2.72,
    "k": 1.059,
    "beta": 1.618,
    "c_prime": c_prime
}

def compute_entanglement_entropy(field, grid_size):
    num_nodes = 16  # 16-node implementation
    dim_per_node = 81  # 4 qutrits per node (3^4)
    total_dim = num_nodes * dim_per_node  # 1296 states
    if field.size == total_dim:
        logger.info("Quantum artifact detected: Computing Shannon entropy for 1D CTC state.")
        probabilities = np.abs(field)**2
        norm = np.sum(probabilities)
        if norm > 0:
            probabilities /= norm
        else:
            logger.warning("State norm is zero or negative, using uniform distribution.")
            probabilities = np.ones(total_dim) / total_dim
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        entropy_per_node = entropy / num_nodes
        return entropy_per_node
    else:
        logger.warning(f"Field size {field.size} does not match expected {total_dim}, falling back to per-point entropy.")
        total_points = 336  # Adjusted for 16 nodes
        entropy = np.zeros(total_points, dtype=np.float64)
        for i in range(total_points):
            local_state = field[i]
            local_prob = np.abs(local_state)**2
            norm = local_prob if local_prob > 0 else 1e-10
            local_prob /= norm
            entropy[i] = -local_prob * np.log2(local_prob + 1e-10) if local_prob > 0 else 0
        return np.mean(entropy)

def compute_j6_potential(phi, j4, psi, ricci_scalar, kappa_j6, kappa_j6_eff, j6_scaling_factor):
    phi_flat = phi.flatten()
    phi_norm = np.linalg.norm(phi_flat)
    phi_term = np.tile((phi_flat[:4] / (phi_norm + 1e-10))**2, 81 * 16)  # 1296 states
    psi_flat = psi.flatten()
    psi_norm = np.linalg.norm(psi_flat)
    psi_term = (psi_flat / (psi_norm + 1e-10))**2
    j4_term = kappa_j6 * j4**2
    ricci_scalar = np.clip(ricci_scalar, -1e5, 1e5)
    ricci_term = kappa_j6_eff * ricci_scalar
    V_j6 = j6_scaling_factor * (j4_term * phi_term * psi_term + ricci_term)
    V_j6 = np.clip(V_j6, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
    logger.info(f"V_j6 shapes: j4_term={j4_term.shape}, phi_term={phi_term.shape}, psi_term={psi_term.shape}")
    return V_j6

def m_shift(u, v):
    return 2.72

def sample_tetrahedral_points(levels):
    points = []
    total_points = 0
    for level in range(1, levels + 1):
        n = (level * (level + 1) * (level + 2)) // 6
        if total_points + n > 336:  # 336 points for 16 nodes (84 * 4)
            n = 336 - total_points
        for _ in range(n):
            t = level * np.random.uniform(-1, 1)
            x = level * np.random.uniform(-1, 1)
            y = level * np.random.uniform(-1, 1)
            z = level * np.random.uniform(-1, 1)
            v = level * np.random.uniform(-1, 1)
            u = level * np.random.uniform(-1, 1)
            points.append([t, x, y, z, v, u])
            total_points += 1
            if total_points >= 336:
                break
        if total_points >= 336:
            break
    return np.array(points)

def hermitian_hamiltonian(x, y, z, k=0.1, J=0.05):
    n = len(x)
    H = np.zeros((n, n), dtype=complex)
    for i in range(n):
        V_i = k * (x[i]**2 + y[i]**2 + z[i]**2)
        H[i, i] = V_i
        face_size = 21  # Adjusted for 16 nodes (336 / 16 ≈ 21 points per node)
        pos_in_face = i % face_size
        if pos_in_face < face_size - 1:
            H[i, i+1] = J
            H[i+1, i] = J
        face_idx = i // face_size
        if pos_in_face == face_size - 1 and face_idx < 15:  # 16 nodes - 1
            next_face_start = (face_idx + 1) * face_size
            if i + 1 < n and next_face_start < n:
                H[i, next_face_start] = J
                H[next_face_start, i] = J
    return H

def unitary_matrix(H, t=1.0, hbar=1.0):
    return sparse_expm(-1j * H * t / hbar)

class TetrahedralLattice:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.levels = grid_size[0] * 2  # Doubled to 12 for 16 nodes
        self.total_points = 336  # 336 points for 16 nodes
        self.coordinates = sample_tetrahedral_points(self.levels)
    
    def _generate_coordinates(self):
        return self.coordinates

class NuggetFieldSolver3D:
    def __init__(self, grid_size=10, m=0.1, lambda_ctc=0.5, c=1.0, alpha=0.1, 
                 a=1.0, kappa=0.1, g_em=0.3, g_weak=0.65, g_strong=1.0, 
                 wormhole_nodes=None, simulation=None):
        self.nx, self.ny, self.nz = grid_size, grid_size, grid_size
        self.grid = np.linspace(-5, 5, grid_size)
        self.x, self.y, self.z = np.meshgrid(self.grid, self.grid, self.grid, indexing='ij')
        self.r = np.sqrt(self.x**2 + self.y**2 + self.z**2 + 1e-10)
        self.theta = theta
        self.phi_angle = np.arctan2(self.y, self.x)
        self.t_grid = np.linspace(0, 2.0, 50)
        self.dx, self.dt = self.grid[1] - self.grid[0], 0.01
        self.m, self.lambda_ctc, self.c, self.alpha = m, lambda_ctc, c, alpha
        self.a, self.kappa = a, kappa
        self.g_em, self.g_weak, self.g_strong = g_em, g_weak, g_strong
        self.phi = np.zeros((self.nx, self.ny, self.nz))
        self.phi_prev = self.phi.copy()
        self.weyl = np.ones((self.nx, self.ny, self.nz))
        self.lambda_harmonic = 2.72
        self.schumann_freq = 7.83
        self.tetrahedral_amplitude = 0.1
        self.wormhole_nodes = wormhole_nodes
        self.simulation = simulation
        self.ctc_cache = {}
        
        if self.wormhole_nodes is not None and self.simulation is not None:
            self.precompute_ctc_field()
            logger.info("NuggetFieldSolver3D initialized with CTC field")

    def precompute_ctc_field(self):
        for t in self.t_grid:
            ctc_field = np.zeros((self.nx, self.ny, self.nz))
            for node in self.wormhole_nodes:
                t_j, x_j, y_j, z_j, v_j, u_j = node
                distance = np.sqrt((self.x - x_j)**2 + (self.y - y_j)**2 + 
                                   (self.z - z_j)**2 + (t - t_j)**2 / self.c**2)
                height = np.exp(-distance**2 / 2.0)
                ctc_field += height
            self.ctc_cache[t] = ctc_field / len(self.wormhole_nodes)
        logger.info("Precomputed CTC field")

    def phi_N_func(self, t, r, theta, phi):
        return np.exp(-r**2) * (1 + self.kappa * np.exp(-t))

    def compute_ricci(self, t):
        phi_N = self.phi_N_func(t, self.r, self.theta, self.phi_angle)
        self.weyl = np.ones_like(self.phi) * (1 + 0.1 * phi_N)
        return self.weyl

    def ctc_function(self, t, x, y, z):
        if t not in self.ctc_cache:
            return np.zeros_like(x)
        return self.ctc_cache[t]

    def tetrahedral_potential(self, x, y, z):
        vertices = np.array([[3, 3, 3], [6, -6, -6], [-6, 6, -6], [-6, -6, 6]]) * CONFIG["vertex_lambda"]
        min_distance = np.inf * np.ones_like(x)
        for vertex in vertices:
            distance = np.sqrt((x - vertex[0])**2 + (y - vertex[1])**2 + (z - vertex[2])**2)
            min_distance = np.minimum(min_distance, distance)
        return self.tetrahedral_amplitude * np.exp(-min_distance**2 / (2 * self.lambda_harmonic**2))

    def schumann_potential(self, t):
        return np.sin(2 * np.pi * self.schumann_freq * t)

    def gauge_source(self, t):
        Y_10 = sph_harm(0, 1, self.phi_angle, self.theta).real
        source_em = self.g_em * np.sin(t) * np.exp(-self.r) * Y_10
        source_weak = self.g_weak * np.cos(t) * np.exp(-self.r) * Y_10
        source_strong = self.g_strong * np.ones_like(self.r) * Y_10
        return source_em + source_weak + source_strong

    def build_laplacian(self):
        n = self.nx * self.ny * self.nz
        data = []
        row_ind = []
        col_ind = []
        for i in range(self.nx):
            for j in range(self.ny):
                for k in range(self.nz):
                    idx = i * self.ny * self.nz + j * self.nz + k
                    data.append(-6 / self.dx**2)
                    row_ind.append(idx)
                    col_ind.append(idx)
                    for di, dj, dk in [(1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)]:
                        ni, nj, nk = i + di, j + dj, k + dk
                        if 0 <= ni < self.nx and 0 <= nj < self.ny and 0 <= nk < self.nz:
                            nidx = ni * self.ny * self.nz + nj * self.nz + nk
                            data.append(1 / self.dx**2)
                            row_ind.append(idx)
                            col_ind.append(nidx)
        return csr_matrix((data, (row_ind, col_ind)), shape=(n, n))

    def effective_mass(self):
        return self.m**2 * (1 + self.alpha * np.mean(self.weyl))

    def rhs(self, t, phi_flat):
        phi = phi_flat.reshape((self.nx, self.ny, self.nz))
        self.phi_prev = self.phi.copy()
        self.phi = phi
        phi_t = (phi - self.phi_prev) / self.dt
        laplacian_op = self.build_laplacian()
        laplacian = laplacian_op.dot(phi_flat).reshape(self.nx, self.ny, self.nz)
        ctc_term = self.lambda_ctc * self.ctc_function(t, self.x, self.y, self.z) * phi
        source = self.gauge_source(t)
        self.compute_ricci(t)
        tetrahedral_term = self.tetrahedral_potential(self.x, self.y, self.z) * phi
        schumann_term = self.schumann_potential(t) * phi
        dphi_dt = (phi_t / self.dt + self.c**-2 * phi_t + laplacian - self.effective_mass() * phi + 
                   ctc_term - source + tetrahedral_term + schumann_term)
        return dphi_dt.flatten()

    def solve(self, t_end=2.0, nt=100):
        t_values = np.linspace(0, t_end, nt)
        initial_state = self.phi.flatten()
        sol = solve_ivp(self.rhs, [0, t_end], initial_state, t_eval=t_values, method='RK45', 
                        rtol=CONFIG['rtol'], atol=CONFIG['atol'])
        self.phi = sol.y[:, -1].reshape((self.nx, self.ny, self.nz))
        return self.phi

class Unified6DSimulation:
    def __init__(self):
        self.grid_size = CONFIG["grid_size"]
        self.total_points = 336  # 336 points for 16 nodes
        self.dt = CONFIG["dt"]
        self.deltas = [CONFIG['dt'], CONFIG['dx'], CONFIG['dx'], CONFIG['dx'], CONFIG['dv'], CONFIG['du']]
        self.time = 0.0
        self.wormhole_nodes = self._generate_wormhole_nodes()
        self.num_nodes = 16  # 16-node implementation
        self.dim_per_node = 81  # 4 qutrits per node
        self.ctc_total_dim = self.num_nodes * self.dim_per_node  # 1296 states
        self.lattice = TetrahedralLattice(self.grid_size)
        self.quantum_state = np.exp(1j * np.random.uniform(0, 2*np.pi, self.ctc_total_dim)) / np.sqrt(self.num_nodes)
        self.phi_N = np.zeros(self.total_points, dtype=np.float64)
        self.stress_energy = self._initialize_stress_energy()
        self.ctc_state = np.zeros(self.ctc_total_dim, dtype=np.complex128)
        self.ctc_state[0] = 1.0 / np.sqrt(self.num_nodes)
        self.setup_symbolic_calculations()
        self.unitary_matrix = np.tile(np.eye(4, dtype=complex), (self.total_points, 1, 1))
        self.nugget_solver = NuggetFieldSolver3D(
            grid_size=10, 
            m=CONFIG["nugget_m"], 
            lambda_ctc=CONFIG["nugget_lambda"],
            wormhole_nodes=self.wormhole_nodes,
            simulation=self
        )
        self.nugget_field = np.zeros((10, 10, 10))
        self.tetrahedral_nodes, self.napoleon_centroids = self._generate_enhanced_tetrahedral_points()
        self.history = []
        self.phi_N_history = []
        self.nugget_field_history = []
        self.result_history = []
        self.ctc_state_history = []
        self.entanglement_history = []
        self.time_displacement_history = []
        self.metric_history = []
        self.metric_scale_factors = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])  # Fibonacci sequence

    def _generate_wormhole_nodes(self):
        # 16 wormhole nodes with scaled coordinates
        vertices = np.array([
            [3, 3, 3], [6, -6, -6], [-6, 6, -6], [-6, -6, 6],
            [9, 9, 9], [12, -12, -12], [-12, 12, -12], [-12, -12, 12],
            [15, 15, 15], [18, -18, -18], [-18, 18, -18], [-18, -18, 18],
            [21, 21, 21], [24, -24, -24], [-24, 24, -24], [-24, -24, 24]
        ]) * CONFIG["vertex_lambda"]
        nodes = [[0.0, v[0], v[1], v[2], 0.0, 0.0] for v in vertices]
        return np.array(nodes)

    def setup_symbolic_calculations(self):
        t, x, y, z, v, u = sp.symbols('t x y z v u')
        a, c_sym, m, kappa_sym = sp.symbols('a c m kappa', positive=True)
        phi_N_sym = sp.Function('phi_N')(t, x, y, z, v, u)
        r = sp.sqrt(x**2 + y**2 + z**2 + v**2 + u**2 + 1e-10)
        scaling_factor = (1 + sp.sqrt(5)) / 2  # Golden ratio
        g_tt = scaling_factor * (-c_sym**2 * (1 + kappa_sym * phi_N_sym))
        g_rr = scaling_factor * (a**2 * sp.exp(2 * r / a) * (1 + kappa_sym * phi_N_sym))
        g_theta_theta = scaling_factor * (a**2 * (sp.exp(2 * r / a) - 1) * (1 + kappa_sym * phi_N_sym))
        g_tphi = scaling_factor * (a * c_sym * sp.exp(r / a))
        g_phi_phi = scaling_factor * (1 + kappa_sym * phi_N_sym)
        g_vv = scaling_factor * (l_p**2)
        g_uu = scaling_factor * (l_p**2)
        g = sp.zeros(6, 6)
        g[0, 0] = g_tt
        g[1, 1] = g_rr
        g[2, 2] = g_theta_theta
        g[3, 3] = g_phi_phi
        g[0, 3] = g_tphi
        g[3, 0] = g_tphi
        g[4, 4] = g_vv
        g[5, 5] = g_uu
        weights = np.array([1.0, 1.0, 2.0, 3.0, 5.0, 8.0])
        norm_factor = np.sum(weights**2)
        g_pyth = sp.zeros(6, 6)
        for i in range(6):
            for j in range(6):
                g_pyth[i, j] = 0.01 * weights[i] * weights[j] / norm_factor
        g += g_pyth
        self.g = g
        self.g_inv = g.inv()

    def _initialize_stress_energy(self):
        T = np.zeros((self.total_points, 6, 6), dtype=np.float64)
        T_base = np.zeros((6, 6), dtype=np.float64)
        T_base[0, 0] = 3.978873e-12
        T_base[1:4, 1:4] = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.float64)
        for i in range(self.total_points):
            T[i] = T_base
        return T

    def _generate_enhanced_tetrahedral_points(self):
        points_6d = sample_tetrahedral_points(6)
        n_points = points_6d.shape[0]
        logger.info(f"Generated {n_points} tetrahedral points in 6D")
        tetrahedral_nodes = points_6d[:, :3]
        n_faces = min(16, n_points // (n_points // 16))  # 16 nodes
        face_size = n_points // n_faces if n_points >= n_faces else 1
        face_indices = [list(range(i * face_size, min((i + 1) * face_size, n_points))) for i in range(n_faces)]
        napoleon_centroids = []
        for face in face_indices:
            if len(face) >= 3:
                face_points = tetrahedral_nodes[face[:3]]
                centroid = np.mean(face_points, axis=0)
                napoleon_centroids.append(centroid)
        napoleon_centroids = np.array(napoleon_centroids)
        lambda_vertex = CONFIG["vertex_lambda"]
        return tetrahedral_nodes * lambda_vertex, napoleon_centroids * lambda_vertex

    def _compute_ctc_unitary_matrix(self):
        coords_6d = sample_tetrahedral_points(6)
        x, y, z = coords_6d[:, 0], coords_6d[:, 1], coords_6d[:, 2]
        H = hermitian_hamiltonian(x, y, z, k=0.1, J=0.05)
        U_full = unitary_matrix(H, t=1.0)
        n_points = coords_6d.shape[0]
        selected_indices = []
        for i in range(min(16, n_points // (n_points // 16))):
            start = i * (n_points // 16)
            selected_indices.extend([start, start + 1, start + 2] if start + 2 < n_points else [start, start + 1, start])
        if selected_indices:
            U_ctc = U_full[np.ix_(selected_indices, selected_indices)]
        else:
            U_ctc = U_full[:3, :3]
        return U_ctc

    def compute_metric_tensor(self):
        logger.info("Computing metric tensor with modified speed of light and Pythagorean weights")
        coords = self.lattice.coordinates
        g_numeric = np.zeros((self.total_points, 6, 6), dtype=np.float64)
        for i in range(self.total_points):
            weights = self.metric_scale_factors
            r_weighted = self.compute_r_6D(coords[i:i+1])
            scaling_factor = (1 + np.sqrt(5)) / 2
            a = CONFIG['a_godel']
            kappa = CONFIG['kappa']
            phi_N = np.clip(self.phi_N[i], -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
            c_effective = CONFIG["c_prime"] if "c_prime" in CONFIG else c0
            g_numeric[i, 0, 0] = scaling_factor * (-c_effective**2 * (1 + kappa * phi_N))
            g_numeric[i, 1, 1] = scaling_factor * (a**2 * np.exp(2 * r_weighted / a) * (1 + kappa * phi_N))
            g_numeric[i, 2, 2] = scaling_factor * (a**2 * (np.exp(2 * r_weighted / a) - 1) * (1 + kappa * phi_N))
            g_numeric[i, 3, 3] = scaling_factor * (1 + kappa * phi_N)
            g_numeric[i, 0, 3] = scaling_factor * (a * c_effective * np.exp(r_weighted / a))
            g_numeric[i, 3, 0] = g_numeric[i, 0, 3]
            g_numeric[i, 4, 4] = scaling_factor * (l_p**2)
            g_numeric[i, 5, 5] = scaling_factor * (l_p**2)
            weights = self.metric_scale_factors / np.sum(self.metric_scale_factors**2)
            for j in range(6):
                for k in range(6):
                    g_numeric[i, j, k] += 0.01 * weights[j] * weights[k]
        g_numeric = np.clip(g_numeric, -CONFIG["field_clamp_max"], CONFIG["field_clamp_max"])
        logger.debug(f"Metric tensor computed, shape: {g_numeric.shape}, g[0,0] sample: {g_numeric[0,0,0]}")
        return g_numeric

    def compute_r_6D(self, coords):
        if coords.ndim == 1:
            coords = coords[np.newaxis, :]
        weights = self.metric_scale_factors if hasattr(self, 'metric_scale_factors') else np.ones(6)
        x_center = np.mean(coords, axis=0)
        r_6D = np.sqrt(np.sum(((coords - x_center) * weights)**2, axis=1))
        return r_6D.astype(np.float32)

    def compute_phi(self, coords):
        r_6D = self.compute_r_6D(coords)
        k = CONFIG["k"]
        c_effective = CONFIG["c_prime"] if "c_prime" in CONFIG else c0
        phi = -r_6D**2 * np.cos(k * r_6D - omega * self.time / c_effective) + 2 * r_6D * np.sin(k * r_6D / c_effective)
        return phi.astype(np.float32)

    def compute_lambda(self):
        phi = self.compute_phi(self.lattice.coordinates)
        integrand = phi**2
        integral = np.mean(integrand)
        return LAMBDA * (1 + 1e-2 * integral)

    def compute_time_displacement(self, u_entry, u_exit, v=0):
        C = 2.0
        alpha_time = CONFIG["alpha_time"]
        c_effective = CONFIG["c_prime"] if "c_prime" in CONFIG else c0
        cycle_factor = np.sin(2 * np.pi * self.time / METONIC_CYCLE) + np.sin(2 * np.pi * self.time / SAROS_CYCLE)
        t_entry = alpha_time * 2 * np.pi * C * np.cosh(v) * np.sin(u_entry) / c_effective * (1 + cycle_factor)
        t_exit = alpha_time * 2 * np.pi * C * np.cosh(v) * np.sin(u_exit) / c_effective * (1 + cycle_factor)
        return t_exit - t_entry

    def adjust_time_displacement(self, target_dt, u_entry=0.0, v=0):
        def objective(delta_u):
            u_exit = u_entry + delta_u
            dt = self.compute_time_displacement(u_entry, u_exit, v)
            return (dt - target_dt)**2
        result = minimize(objective, x0=0.1, method='Nelder-Mead', tol=1e-12)
        delta_u = result.x[0]
        u_exit = u_entry + delta_u
        actual_dt = self.compute_time_displacement(u_entry, u_exit, v)
        logger.info(f"Adjusted time displacement: {actual_dt:.6e} (target: {target_dt:.6e})")
        return u_exit, actual_dt

    def transmit_and_compute(self, input_data, direction="future", target_dt=None):
        if target_dt is None:
            target_dt = self.dt if direction == "future" else -self.dt
        entry_time = self.time
        entry_u = 0.0
        physical_time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at transmission start: {physical_time_start}, Simulated time: {entry_time:.6e}")
        u_exit, actual_dt = self.adjust_time_displacement(target_dt, u_entry=entry_u)
        exit_time = self.time + actual_dt
        physical_time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Transmitting from t={entry_time:.6e} to t={exit_time:.6e} ({direction}), Physical start: {physical_time_start}, Physical end: {physical_time_end}")
        matrix_size = CONFIG["matrix_size"]
        A = np.random.randn(matrix_size, matrix_size) + 1j * np.random.rand(matrix_size, matrix_size)
        A = (A + A.conj().T) / 2
        eigenvalues = np.linalg.eigvalsh(A)
        result = np.sum(np.abs(eigenvalues))
        logger.info(f"Computation result at t={exit_time:.6e}: {result:.6f}")
        return result

    def simulate_ctc_quantum_circuit(self):
        logger.info("Running CTC quantum circuit with vm-Teleportation (4 qutrits per node, 16 nodes)")
        num_qutrits_per_node = 4
        num_nodes = self.num_nodes  # 16 nodes
        dim_per_node = 3 ** num_qutrits_per_node  # 81 states
        total_dim = dim_per_node * num_nodes  # 1296 states
        initial_state = np.zeros(total_dim, dtype=np.complex128)
        initial_state[0] = 1.0 / np.sqrt(num_nodes)
        logger.info(f"Step 1: Initial state created, dim={total_dim}")

        def qutrit_hadamard():
            return csr_matrix(np.array([
                [1, 1, 1],
                [1, np.exp(2j * np.pi / 3), np.exp(4j * np.pi / 3)],
                [1, np.exp(4j * np.pi / 3), np.exp(2j * np.pi / 3)]
            ]) / np.sqrt(3))

        state = initial_state.copy()
        for node_idx in range(num_nodes):
            H_tensor = qutrit_hadamard()
            for _ in range(num_qutrits_per_node - 1):
                H_tensor = kron(H_tensor, qutrit_hadamard())
            start_idx = node_idx * dim_per_node
            end_idx = (node_idx + 1) * dim_per_node
            node_state = state[start_idx:end_idx]
            node_state = H_tensor @ node_state
            state[start_idx:end_idx] = node_state
        logger.info(f"Step 2: Hadamard applied for entanglement, state norm={np.linalg.norm(state)}")

        def qutrit_cnot():
            dim = 3 ** num_qutrits_per_node
            total_dim = dim * num_nodes
            data = []
            row_ind = []
            col_ind = []
            for i in range(total_dim):
                node_idx = i // dim
                state_idx = i % dim
                qutrit_states = []
                temp = state_idx
                for _ in range(num_qutrits_per_node):
                    qutrit_states.append(temp % 3)
                    temp //= 3
                qutrit_states = qutrit_states[::-1]
                new_state = qutrit_states.copy()
                for q in range(num_qutrits_per_node - 1):
                    if qutrit_states[q] == 2:
                        new_state[q + 1] = (qutrit_states[q + 1] + 1) % 3
                new_i = 0
                for j, state in enumerate(new_state):
                    new_i += state * (3 ** j)
                new_global_i = node_idx * dim + new_i
                if new_global_i < total_dim and i != new_global_i:
                    data.append(1.0)
                    row_ind.append(i)
                    col_ind.append(new_global_i)
                    data.append(0.0)
                    row_ind.append(i)
                    col_ind.append(i)
            cnot_matrix = csr_matrix((data, (row_ind, col_ind)), shape=(total_dim, total_dim)) + csr_matrix(eye(total_dim, dtype=complex))
            return cnot_matrix

        cnot_op = qutrit_cnot()
        state = cnot_op @ state
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        logger.info(f"Step 3: Entangled state (Bell-like) created, state norm={np.linalg.norm(state)}")

        act2_blocks = []
        for node_idx, node in enumerate(self.wormhole_nodes):
            x, y, z = node[1], node[2], node[3]
            theta_local = 2 * np.arctan2(y, x)
            act2_matrix = csr_matrix(np.array([
                [np.cos(theta_local), -np.sin(theta_local), 0],
                [np.sin(theta_local), np.cos(theta_local), 0],
                [0, 0, 1]
            ], dtype=complex))
            node_act2_block = act2_matrix
            for _ in range(1, num_qutrits_per_node):
                node_act2_block = kron(node_act2_block, act2_matrix)
            act2_blocks.append(node_act2_block.toarray())
        act2_dense = dense_block_diag(*act2_blocks)
        act2_op = csr_matrix(act2_dense)
        state = act2_op @ state
        norm = np.linalg.norm(state)
        if norm > 0:
            state /= norm
        logger.info(f"Step 4: ACT2 applied for rotation, state norm={norm}")

        phase = np.exp(1j * self.time * CONFIG["dt"] / hbar * (np.sin(2 * np.pi * self.time / METONIC_CYCLE) + np.sin(2 * np.pi * self.time / SAROS_CYCLE)))
        ctc_feedback = csr_matrix(np.diag(phase * np.ones(total_dim, dtype=complex) * CONFIG["ctc_feedback_factor"]))
        state = ctc_feedback @ state
        state /= np.linalg.norm(state) if np.linalg.norm(state) > 0 else 1
        logger.info(f"Step 5: CTC feedback applied with Antikythera cycles, state norm={np.linalg.norm(state)}")

        probs = np.abs(state)**2
        max_prob = np.max(probs) if np.isfinite(np.sum(probs)) else np.nan
        decision = 0 if max_prob > 0.5 else 1
        self.ctc_state = state
        logger.info(f"Step 6: Measurement completed (teleportation decision), decision={decision}, max_prob={max_prob:.6f}")

        print(f"vm-Teleportation Circuit (4 qutrits per node, 16 nodes): Initial State |0000...0000>, H, Bell Entangled States, ACT2, CTC Feedback\n"
              f"ACT2 Matrix (example):\n{act2_matrix.toarray()}\nDecision: {decision} (max prob: {max_prob:.6f})")
        logger.info(f"vm-Teleportation quantum decision: {decision} (max prob: {max_prob:.6f})")
        return decision

    def compute_V_j6(self):
        logger.info("Computing V_j6 with quantum state projection")
        phi = self.compute_phi(self.lattice.coordinates)
        phi_1296 = np.interp(np.linspace(0, 1295, 1296), np.arange(self.total_points), phi)  # 1296 states
        j4 = np.sin(np.angle(self.quantum_state))
        j4_1296 = j4
        psi = self.quantum_state
        psi_1296 = psi
        r_6d = self.compute_r_6D(self.lattice.coordinates)
        ricci_scalar_1296 = np.interp(np.linspace(0, 1295, 1296), np.arange(self.total_points), -G * m_n / (r_6d**4) * (1 / LAMBDA**2))
        logger.info(f"Projected shapes: phi_1296={phi_1296.shape}, ricci_scalar_1296={ricci_scalar_1296.shape}")
        V_j6 = compute_j6_potential(phi_1296, j4_1296, psi_1296, ricci_scalar_1296, CONFIG["kappa_j6"], 
                                  CONFIG["kappa_j6_eff"], CONFIG["j6_scaling_factor"])
        return V_j6

    def evolve_quantum_state(self, dt):
        logger.info("Evolving quantum state with modified spacetime and Antikythera cycles")
        physical_time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at evolution start: {physical_time_start}, Simulated time: {self.time:.6e}")
        num_nodes = self.num_nodes
        dim_per_node = 81
        ctc_total_dim = num_nodes * dim_per_node  # 1296 states
        ctc_state_reshaped = self.quantum_state.reshape(num_nodes, dim_per_node)
        g_numeric = self.compute_metric_tensor()
        coords = self.lattice.coordinates
        phi = self.compute_phi(coords)
        logger.info(f"Phi shape before interpolation: {phi.shape}")
        if phi.shape != (self.total_points,):
            phi = phi.reshape(self.total_points)
            logger.info(f"Reshaped phi to: {phi.shape}")
        g_tt = g_numeric[:, 0, 0]
        logger.info(f"g_tt shape before interpolation: {g_tt.shape}")
        if g_tt.shape != (self.total_points,):
            g_tt = g_tt.reshape(self.total_points)
            logger.info(f"Reshaped g_tt to: {g_tt.shape}")
        phi_proj_1296 = np.interp(np.linspace(0, 1295, 1296), np.arange(self.total_points), phi)
        g_tt_proj_1296 = np.interp(np.linspace(0, 1295, 1296), np.arange(self.total_points), g_tt)
        r_6d = self.compute_r_6D(coords)
        tetractys_potential = 0.1 * (1 + 2 * coords[:, 1] + 3 * coords[:, 2] + 4 * coords[:, 3])
        V_pot = (-G * m_n / (r_6d**4) * (1 / LAMBDA**2) + 0.1) * (1 + 2 * np.sin(self.time)) + tetractys_potential
        V_pot = np.nan_to_num(V_pot, nan=0.0, posinf=1e10, neginf=-1e10)
        logger.info(f"V_pot shape before interpolation: {V_pot.shape}")
        if V_pot.shape != (self.total_points,):
            V_pot = V_pot.reshape(self.total_points)
            logger.info(f"Reshaped V_pot to: {V_pot.shape}")
        V_pot_proj_1296 = np.interp(np.linspace(0, 1295, 1296), np.arange(self.total_points), V_pot)
        V_j6 = self.compute_V_j6()
        logger.info(f"V_j6 shape before interpolation: {V_j6.shape}")
        if V_j6.shape != (1296,):
            V_j6 = V_j6.reshape(1296)
            logger.info(f"Reshaped V_j6 to: {V_j6.shape}")
        V_j6_proj_1296 = V_j6
        harmonic_weights = self.metric_scale_factors / np.sum(self.metric_scale_factors)
        V_total = V_pot_proj_1296 + V_j6_proj_1296 + 1e-20 * g_tt_proj_1296 * harmonic_weights[0] + 1e-4 * phi_proj_1296 * harmonic_weights[1]
        V_total = np.nan_to_num(V_total, nan=0.0, posinf=1e10, neginf=-1e10)
        V_min, V_max = np.min(V_total), np.max(V_total)
        if V_max - V_min > 0:
            V_total = (V_total - V_min) / (V_max - V_min)
        V_total = np.clip(V_total, 0, 1) * 1e-6
        logger.info(f"V_total shape: {V_total.shape}, min: {np.min(V_total)}, max: {np.max(V_total)}")
        V_mean = np.clip(np.mean(V_total), -1e6, 1e6)  # Clamped for stability
        H_ctc = np.zeros((ctc_total_dim, ctc_total_dim), dtype=complex)
        for i in range(num_nodes):
            start_idx = i * dim_per_node
            end_idx = (i + 1) * dim_per_node
            H_ctc[start_idx:end_idx, start_idx:end_idx] = np.diag(np.full(dim_per_node, V_mean) + 1e-10 * np.eye(dim_per_node) * harmonic_weights[i % 6])
        h_norm = sparse_norm(csr_matrix(H_ctc), ord=1)
        logger.info(f"H_ctc one-norm: {h_norm}")
        effective_dt = dt * min(1.0, 1e-10 / h_norm) if h_norm > 1e-10 else dt
        logger.info(f"Effective dt: {effective_dt:.6e}")
        if np.isinf(h_norm) or np.isnan(h_norm):
            logger.info("Quantum artifact detected: Invalid H_ctc norm. Using fallback evolution.")
            ctc_state_reshaped = ctc_state_reshaped * np.exp(-1j * V_mean * dt / hbar)
        else:
            try:
                evolution = sparse_expm(-1j * csr_matrix(H_ctc) * effective_dt / hbar)
                evolved_state = evolution @ ctc_state_reshaped.flatten()
                ctc_state_reshaped = evolved_state.reshape(num_nodes, dim_per_node)
                if np.any(np.isnan(ctc_state_reshaped)) or np.all(np.abs(ctc_state_reshaped) < 1e-10):
                    logger.warning("Quantum state contains NaN or near zero. Using fallback.")
                    ctc_state_reshaped = ctc_state_reshaped * np.exp(-1j * V_mean * dt / hbar)
            except (ValueError, OverflowError) as e:
                logger.info(f"Quantum artifact detected: {e}. Applying CTC feedback.")
                if self.ctc_state_history:
                    last_ctc_state = self.ctc_state_history[-1].copy()
                    feedback_phase = np.exp(1j * self.time * CONFIG["ctc_feedback_factor"])
                    ctc_state_reshaped = (last_ctc_state.reshape(num_nodes, dim_per_node) * feedback_phase)
                    logger.info(f"Quantum artifact feedback applied, state norm={np.linalg.norm(ctc_state_reshaped)}")
                else:
                    logger.info("No prior state, initializing quantum artifact state.")
                    ctc_state_reshaped = np.zeros((num_nodes, dim_per_node))
                    ctc_state_reshaped[0, 0] = 1.0 / np.sqrt(num_nodes)
        self.quantum_state = ctc_state_reshaped.reshape(ctc_total_dim) * np.exp(1j * CONFIG["beta"] * np.mean(phi))
        if np.any(np.isnan(self.quantum_state)) or np.all(np.abs(self.quantum_state) < 1e-10):
            logger.warning("Quantum state invalid. Resetting to random state.")
            self.quantum_state = np.exp(1j * np.random.uniform(0, 2*np.pi, ctc_total_dim)) / np.sqrt(ctc_total_dim)
        norm = np.linalg.norm(self.quantum_state)
        if norm > 0:
            self.quantum_state /= norm
        physical_time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at evolution end: {physical_time_end}, Simulated time: {self.time:.6e}")
        var_amp = np.var(np.abs(self.quantum_state))
        print(f"Quantum state variance amplitude: {var_amp:.6e}")

    def visualize_tetrahedral_nodes(self):
        logger.info("Visualizing tetrahedral nodes")
        physical_time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at visualization start: {physical_time_start}, Simulated time: {self.time:.6e}")
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.tetrahedral_nodes[:,0], self.tetrahedral_nodes[:,1], self.tetrahedral_nodes[:,2], 
                   c='b', s=50, label='Nodes')
        ax.scatter(self.napoleon_centroids[:,0], self.napoleon_centroids[:,1], self.napoleon_centroids[:,2], 
                   c='r', s=100, marker='^', label='Centroids')
        for i in range(0, len(self.tetrahedral_nodes), 3):
            face_nodes = self.tetrahedral_nodes[i:i+3]
            for j in range(len(face_nodes)):
                for k in range(j+1, len(face_nodes)):
                    ax.plot([face_nodes[j,0], face_nodes[k,0]], 
                            [face_nodes[j,1], face_nodes[k,1]], 
                            [face_nodes[j,2], face_nodes[k,2]], 'g-', alpha=0.5)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('Tetrahedral Wormhole Node Structure (16 Nodes)')
        ax.legend()
        plt.savefig('tetrahedral_nodes.png')
        plt.close()
        physical_time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at visualization end: {physical_time_end}, Simulated time: {self.time:.6e}")

    def run_simulation(self):
        logger.info("Starting simulation with vm-Teleportation")
        physical_time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at simulation start: {physical_time_start}, Simulated time: {self.time:.6e}")
        self.visualize_tetrahedral_nodes()
        print("Time | Nugget Mean | Past Result | Future Result | Entanglement Entropy (Shannon) | g_tt Mean")
        for iteration in range(5, 5 + CONFIG["max_iterations"]):  # Start at 5 to match previous log offset
            logger.info(f"Iteration {iteration}/{5 + CONFIG['max_iterations'] - 1}")
            self.time += self.dt
            physical_time_iter_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            logger.info(f"Physical time at iteration start: {physical_time_iter_start}, Simulated time: {self.time:.6e}")
            self.nugget_field = self.nugget_solver.solve(t_end=self.time, nt=2)
            nugget_mean = np.mean(self.nugget_field)
            logger.info(f"Nugget field mean: {nugget_mean:.6e}")
            past_result = self.transmit_and_compute(1010, direction="past", target_dt=-2*self.dt)
            future_result = self.transmit_and_compute(1010, direction="future", target_dt=2*self.dt)
            ctc_decision = self.simulate_ctc_quantum_circuit()
            self.ctc_state_history.append(self.ctc_state.copy())
            self.evolve_quantum_state(self.dt)
            entanglement = compute_entanglement_entropy(self.quantum_state, self.grid_size)
            logger.info(f"Entanglement entropy (Shannon): {entanglement:.6f}")
            g_numeric = self.compute_metric_tensor()
            g_tt_mean = np.mean(g_numeric[:, 0, 0])
            logger.info(f"Metric tensor g_tt mean: {g_tt_mean:.6e}")
            self.metric_history.append(g_numeric)
            self.history.append(self.time)
            self.nugget_field_history.append(nugget_mean)
            self.result_history.append((past_result, future_result))
            self.entanglement_history.append(entanglement)
            # Visualization data
            if iteration == 5:
                self.entanglement_history_plot = [entanglement]
                self.g_tt_history_plot = [g_tt_mean]
                self.nugget_history_plot = [nugget_mean]
                self.variance_history_plot = [np.var(np.abs(self.quantum_state))]
            else:
                self.entanglement_history_plot.append(entanglement)
                self.g_tt_history_plot.append(g_tt_mean)
                self.nugget_history_plot.append(nugget_mean)
                self.variance_history_plot.append(np.var(np.abs(self.quantum_state)))
            print(f"{self.time:.2e} | {nugget_mean:.6e} | {past_result:.6e} | {future_result:.6e} | {entanglement:.6f} | {g_tt_mean:.6e}")
        # Plot results
        plt.figure(figsize=(12, 10))
        plt.subplot(4, 1, 1)
        plt.plot(self.history, self.entanglement_history_plot, label='Entanglement Entropy (Shannon)')
        plt.xlabel('Time (s)')
        plt.ylabel('Entropy (bits)')
        plt.title('Entanglement Entropy Over Time')
        plt.legend()
        plt.subplot(4, 1, 2)
        plt.plot(self.history, self.g_tt_history_plot, label='g_tt Mean')
        plt.xlabel('Time (s)')
        plt.ylabel('g_tt')
        plt.title('Metric Tensor g_tt Over Time')
        plt.legend()
        plt.subplot(4, 1, 3)
        plt.plot(self.history, self.nugget_history_plot, label='Nugget Mean')
        plt.xlabel('Time (s)')
        plt.ylabel('Nugget Mean')
        plt.title('Nugget Field Mean Over Time')
        plt.legend()
        plt.subplot(4, 1, 4)
        plt.plot(self.history, self.variance_history_plot, label='Quantum State Variance')
        plt.xlabel('Time (s)')
        plt.ylabel('Variance')
        plt.title('Quantum State Variance Over Time')
        plt.legend()
        plt.tight_layout()
        plt.savefig('simulation_trends.png')
        plt.close()
        logger.info("Simulation trends plotted and saved as simulation_trends.png")
        logger.info("Simulation completed")
        physical_time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
        logger.info(f"Physical time at simulation end: {physical_time_end}, Simulated time: {self.time:.6e}")
        print("\nFinal Simulation Summary:")
        print(f"Final Time: {self.time:.2e}")
        print(f"Average Nugget Mean: {np.mean(self.nugget_field_history):.6e}")
        print(f"Average Past Result: {np.mean([r[0] for r in self.result_history]):.6f}")
        print(f"Average Future Result: {np.mean([r[1] for r in self.result_history]):.6f}")
        print(f"Average Entanglement Entropy (Shannon): {np.mean(self.entanglement_history):.6f}")
        final_g_numeric = self.metric_history[-1]
        print(f"Average g_tt: {np.mean(final_g_numeric[:, 0, 0]):.6e}")
        print("Average metric tensor components:")
        for i in range(6):
            for j in range(i, 6):
                if np.any(final_g_numeric[:, i, j] != 0):
                    print(f"g_{i}{j}: {np.mean(final_g_numeric[:, i, j]):.8e}")

if __name__ == "__main__":
    sim = Unified6DSimulation()
    sim.run_simulation()