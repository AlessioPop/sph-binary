#!/usr/bin/env python3
"""
2D SPH Binary Simulation
========================
General-purpose 2D SPH simulation of a circum-primary gas ring in a
binary system.  The central star M1 is orbited by a companion M2 on a
wide eccentric orbit.  SPH particles form a ring perturbed by the
companion at periastron.

Includes adaptive smoothing length, adaptive CFL timestep, and
Monaghan (1997) signal-velocity viscosity.

Produces:
  - sph_simulation.mp4   (animation)
  - time_series.png      (mean-radius diagnostic)
  - diagnostics.log      (full parameter + runtime log)
"""

import os, sys, time, json, datetime, math
import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable

# ======================================================================
#  DEFAULT PARAMETERS  (all in code units: au, M_sun, yr)
# ======================================================================
DEFAULTS = {
    # Physical system (binary: central star + companion)
    "M1":        10.29,          # Central star mass [Msun]
    "M2":        4.5,            # Companion mass [Msun]
    "a2":        19.8,           # Companion semi-major axis [au]
    "e2":        0.98,           # Companion eccentricity
    "CS":        1.056,          # Isothermal sound speed [au/yr]
    # SPH kernel & smoothing
    "eta":       1.2,            # Smoothing-length parameter
    "h_min":     1e-4,           # Minimum smoothing length [au]
    "h_max":     0.03,           # Maximum smoothing length [au]
    "h_iter":    3,              # Smoothing-length iterations per step
    # Artificial viscosity
    "alpha_visc":0.1,            # Viscosity linear coefficient
    "beta_visc": 0.2,            # Viscosity quadratic coefficient
    # Timestep
    "gamma_CFL": 0.3,            # CFL safety factor
    "eps_acc":   1e-4,           # Acceleration softening [au/yr^2]
    "dt_min":    1e-7,           # Minimum timestep [yr]
    # Injection / removal
    "m_sph":     1e-6,           # SPH particle mass [Msun] (auto-synced)
    "r_remove_out": 2.0,         # Outer removal radius [au]
    "r_remove_in":  0.05,        # Inner removal radius [au]
    "rho_ref":   1e-2,           # Reference density for new particles
    "N_dot":     0.0,            # Injection rate [yr^-1] (0 = no injection)
    "r_inj":     0.15,           # Injection radius [au]
    "v_inj":     0.0,            # Injection velocity [au/yr] (0 = local Keplerian)
    # Gravity softening
    "eps_grav":  1e-4,           # Gravitational softening [au]
    # Simulation
    "t_end":     .4,            # Total simulation time [yr]
    "dt_snap":   0.05,           # Snapshot interval [yr]
    "n_frames":  300,            # Total animation frames
    "fps":       10,             # Animation frames per second
    "num_threads": 0,            # Numba worker threads (0 = use default/all)
    "plot_lim":  1.5,            # Half-width of animation x/y axes [au]
    "N_seed":    1000,           # Initial ring particles (pre-existing ring)
    "M_disk_seed": 1e-3,         # Total mass of the seeded ring [Msun]
    "t_peri":    0.2,            # Periastron time [yr] (set 0 for auto: 0.5*P2)
    "render_grid": 256,          # Grid resolution for Gaussian density rendering
}

def sync_seed_mass_params(params, changed_key=None):
    """Keep N_seed, M_disk_seed, and m_sph mutually consistent."""
    params["N_seed"] = max(int(params.get("N_seed", 0)), 0)
    params["M_disk_seed"] = max(float(params.get("M_disk_seed", 0.0)), 0.0)
    params["m_sph"] = max(float(params.get("m_sph", 0.0)), 0.0)

    if changed_key == "m_sph":
        params["M_disk_seed"] = params["N_seed"] * params["m_sph"]
        return (f"auto: M_disk_seed = {params['M_disk_seed']:.6g} Msun "
                f"for N_seed = {params['N_seed']}")

    if params["N_seed"] > 0:
        params["m_sph"] = params["M_disk_seed"] / params["N_seed"]
        if changed_key in {"N_seed", "M_disk_seed"}:
            return (f"auto: m_sph = {params['m_sph']:.6g} Msun "
                    f"from M_disk_seed / N_seed")

    return None


def configure_numba_threads(params):
    """Configure Numba worker threads. 0 keeps the default pool size."""
    requested = max(int(params.get("num_threads", 0)), 0)
    params["num_threads"] = requested
    if requested > 0:
        set_num_threads(requested)
    return get_num_threads()


# ======================================================================
#  INTERACTIVE PARAMETER PANEL
# ======================================================================
def parameter_panel(params):
    """Display all parameters and let the user change any before running."""
    sync_seed_mass_params(params)
    print("\n" + "=" * 70)
    print("   SPH Simulation — Parameter Overview")
    print("=" * 70)

    keys = list(params.keys())
    # Group labels
    groups = {
        "Physical system":  ["M1","M2","a2","e2","CS"],
        "SPH kernel":       ["eta","h_min","h_max","h_iter"],
        "Artificial visc.": ["alpha_visc","beta_visc"],
        "Timestep":         ["gamma_CFL","eps_acc","dt_min"],
        "Injection/removal":["m_sph","r_remove_out","r_remove_in","rho_ref",
                             "N_dot","r_inj","v_inj"],
        "Gravity":          ["eps_grav"],
        "Simulation":       ["t_end","dt_snap","n_frames","fps","num_threads","plot_lim",
                             "N_seed","M_disk_seed","t_peri","render_grid"],
    }

    idx = 1
    key_index = {}
    for gname, gkeys in groups.items():
        print(f"\n  --- {gname} ---")
        for k in gkeys:
            v = params[k]
            print(f"  [{idx:2d}] {k:16s} = {v}")
            key_index[idx] = k
            idx += 1

    print("\n" + "-" * 70)
    print("  Press ENTER to accept all defaults and start the simulation.")
    print("  Or type the number of a parameter to change it.")
    print("  Type 'q' when done editing.\n")

    while True:
        try:
            choice = input("  >> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if choice == "" or choice.lower() == "q":
            break
        try:
            num = int(choice)
        except ValueError:
            print("  Invalid input. Enter a number, ENTER, or 'q'.")
            continue
        if num not in key_index:
            print(f"  No parameter #{num}. Valid range: 1-{idx-1}")
            continue
        k = key_index[num]
        current = params[k]
        try:
            raw = input(f"    {k} [{current}] = ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if raw == "":
            print(f"    (kept {current})")
            continue
        # Cast to same type
        try:
            if isinstance(current, int):
                params[k] = int(raw)
            else:
                params[k] = float(raw)
            print(f"    -> {k} = {params[k]}")
            sync_msg = sync_seed_mass_params(params, changed_key=k)
            if sync_msg is not None:
                print(f"    {sync_msg}")
        except ValueError:
            print(f"    Could not parse '{raw}' — kept {current}")

    sync_seed_mass_params(params)
    return params


# ======================================================================
#  DERIVED CONSTANTS (computed from parameter dict)
# ======================================================================
def derive_constants(p):
    """Return a dict of derived quantities used throughout the simulation."""
    G = 4.0 * np.pi**2
    M_tot = p["M1"]
    GM_tot = G * M_tot
    GM2 = G * p["M2"]
    GM_all = G * (p["M1"] + p["M2"])

    P2 = 2.0 * np.pi * np.sqrt(p["a2"]**3 / GM_all)
    t_peri = p["t_peri"] if p.get("t_peri", 0.0) > 0.0 else 0.5 * P2
    N_dot = p["N_dot"]
    # Timestep cap from ring orbital period at injection radius
    r_ref = max(p["r_inj"], 0.05)
    P_ref = 2.0 * np.pi / np.sqrt(GM_tot / r_ref**3)
    dt_max_tidal = P_ref / 50.0
    h_new_default = p["eta"] * np.sqrt(p["m_sph"] / p["rho_ref"])

    return {
        "G": G, "M_tot": M_tot, "GM_tot": GM_tot, "GM2": GM2, "GM_all": GM_all,
        "P2": P2, "t_peri": t_peri,
        "N_dot": N_dot, "dt_max_tidal": dt_max_tidal,
        "h_new_default": h_new_default,
    }


# ======================================================================
#  NUMBA-JIT KERNEL FUNCTIONS
# ======================================================================
@njit(fastmath=True)
def kernel_W(dx, dy, h):
    """M4 cubic spline — returns W only (no gradient, faster for density)."""
    r = math.sqrt(dx * dx + dy * dy)
    if r < 1e-12:
        return 10.0 / (7.0 * math.pi * h * h)
    s = r / h
    if s >= 2.0:
        return 0.0
    norm = 10.0 / (7.0 * math.pi * h * h)
    if s < 1.0:
        return norm * (1.0 - 1.5 * s * s + 0.75 * s * s * s)
    tmp = 2.0 - s
    return norm * 0.25 * tmp * tmp * tmp


@njit(fastmath=True)
def kernel_and_grad(dx, dy, h):
    """M4 cubic spline in 2D. Returns (W, gx, gy)."""
    r = math.sqrt(dx * dx + dy * dy)
    if r < 1e-12:
        return 10.0 / (7.0 * math.pi * h * h), 0.0, 0.0
    s = r / h
    norm = 10.0 / (7.0 * math.pi * h * h)
    if s < 1.0:
        W = norm * (1.0 - 1.5 * s * s + 0.75 * s * s * s)
        dW = norm / h * (-3.0 * s + 2.25 * s * s)
    elif s < 2.0:
        tmp = 2.0 - s
        W = norm * 0.25 * tmp * tmp * tmp
        dW = norm / h * (-0.75 * tmp * tmp)
    else:
        return 0.0, 0.0, 0.0
    gx = dW * dx / r
    gy = dW * dy / r
    return W, gx, gy


@njit(parallel=True, fastmath=True)
def sph_density_neigh(pos, mass, h, row_ptr, col_idx):
    """Compute SPH density using CSR neighbour lists (kernel_W only)."""
    N = pos.shape[0]
    rho = np.zeros(N)
    for i in prange(N):
        hi = h[i]
        rhoi = mass[i] * kernel_W(0.0, 0.0, hi)
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            rhoi += mass[j] * kernel_W(
                pos[i, 0] - pos[j, 0], pos[i, 1] - pos[j, 1], hi)
        rho[i] = rhoi
    return rho


@njit(parallel=True, fastmath=True)
def sph_forces_neigh(pos, vel, mass, rho, P_over_rho2, h,
                     alpha, beta, cs, row_ptr, col_idx):
    """SPH pressure + Monaghan 1997 viscosity using CSR neighbour lists.

    P_over_rho2 = cs^2 / rho  is pre-computed by the caller to avoid
    redundant work inside the inner loop.
    """
    N = pos.shape[0]
    acc = np.zeros((N, 2))
    dv = np.zeros(N)
    for i in prange(N):
        hi = h[i]
        pterm_i = P_over_rho2[i]
        inv_rho_i = 1.0 / rho[i]
        rho_i = rho[i]
        ax_i = 0.0
        ay_i = 0.0
        dv_i = 0.0
        cutoff2 = (2.0 * hi) * (2.0 * hi)
        for k in range(row_ptr[i], row_ptr[i + 1]):
            j = col_idx[k]
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r2 = dx * dx + dy * dy
            if r2 > cutoff2 or r2 < 1e-24:
                continue
            W, gx, gy = kernel_and_grad(dx, dy, hi)
            # Pressure (pre-computed P/rho^2)
            fac = mass[j] * (pterm_i + P_over_rho2[j])
            ax_i -= fac * gx
            ay_i -= fac * gy
            # Velocity differences — computed once, reused below
            vij_x = vel[i, 0] - vel[j, 0]
            vij_y = vel[i, 1] - vel[j, 1]
            # Velocity divergence
            dv_i += mass[j] * inv_rho_i * (vij_x * gx + vij_y * gy)
            # Monaghan 1997 viscosity (only for approaching pairs)
            vdotr = vij_x * dx + vij_y * dy
            if vdotr < 0.0:
                rij = math.sqrt(r2)
                vsig = 2.0 * cs - beta * vdotr / rij
                rhoij = 0.5 * (rho_i + rho[j])
                Pi_ij = -alpha * vsig * vdotr / (rhoij * rij)
                ax_i -= mass[j] * Pi_ij * gx
                ay_i -= mass[j] * Pi_ij * gy
        acc[i, 0] = ax_i
        acc[i, 1] = ay_i
        dv[i] = dv_i
    return acc, dv


# ======================================================================
#  PURE-PYTHON HELPER FUNCTIONS
# ======================================================================
def solve_kepler(M_anom, ecc, tol=1e-10):
    """Newton-Raphson Kepler solver.  M_anom can be scalar or array."""
    M_anom = np.atleast_1d(np.asarray(M_anom, dtype=np.float64))
    E = M_anom.copy()
    for _ in range(50):
        dE = (E - ecc * np.sin(E) - M_anom) / (1.0 - ecc * np.cos(E))
        E -= dE
        if np.all(np.abs(dE) < tol):
            break
    return E


def companion_position(t, c):
    """Return (x2, y2) of the companion at time t [yr]."""
    M = 2.0 * np.pi / c["P2"] * (t - c["t_peri"])
    E = solve_kepler(M, c["e2_val"])[0] if np.ndim(M) == 0 else solve_kepler(M, c["e2_val"])
    E = float(E) if np.ndim(E) == 0 else E
    x2 = c["a2_val"] * (np.cos(E) - c["e2_val"])
    y2 = c["a2_val"] * np.sqrt(1.0 - c["e2_val"] ** 2) * np.sin(E)
    return np.array([x2, y2])


def central_gravity(pos, GM_tot, eps):
    """Central gravity acceleration: -GM r / |r|^3 with softening."""
    r2 = pos[:, 0] ** 2 + pos[:, 1] ** 2 + eps ** 2
    r3 = r2 ** 1.5
    acc = np.empty_like(pos)
    acc[:, 0] = -GM_tot * pos[:, 0] / r3
    acc[:, 1] = -GM_tot * pos[:, 1] / r3
    return acc


def tidal_force(pos, r2_pos, GM2, eps):
    """Tidal acceleration from the companion."""
    r2_norm3 = (r2_pos[0] ** 2 + r2_pos[1] ** 2 + eps ** 2) ** 1.5
    diff = r2_pos[np.newaxis, :] - pos          # (N,2)
    dist3 = (diff[:, 0] ** 2 + diff[:, 1] ** 2 + eps ** 2) ** 1.5
    atidal = GM2 * (diff / dist3[:, np.newaxis] - r2_pos / r2_norm3)
    return atidal


# ======================================================================
#  NEIGHBOUR SEARCH  (cKDTree + O(M) Numba CSR build)
# ======================================================================
@njit
def _count_pairs(pairs, N):
    """Count directed edges per particle from symmetric pair list."""
    counts = np.zeros(N, dtype=np.int64)
    M = len(pairs)
    for k in range(M):
        counts[pairs[k, 0]] += 1
        counts[pairs[k, 1]] += 1
    return counts


@njit
def _fill_csr_from_pairs(pairs, row_ptr):
    """Scatter pair list into CSR col_idx in O(M) — no sorting needed."""
    M = len(pairs)
    N = len(row_ptr) - 1
    col_idx = np.empty(2 * M, dtype=np.int64)
    write_pos = np.empty(N, dtype=np.int64)
    for i in range(N):
        write_pos[i] = row_ptr[i]
    for k in range(M):
        i = pairs[k, 0]
        j = pairs[k, 1]
        col_idx[write_pos[i]] = j
        write_pos[i] += 1
        col_idx[write_pos[j]] = i
        write_pos[j] += 1
    return col_idx


def build_neighbour_csr(pos, radius):
    """Build CSR neighbour list — cKDTree query + O(M) Numba scatter."""
    N = len(pos)
    if N == 0:
        return np.zeros(1, dtype=np.int64), np.empty(0, dtype=np.int64)
    tree = cKDTree(pos)
    pairs = tree.query_pairs(radius, output_type='ndarray')
    if len(pairs) == 0:
        return np.zeros(N + 1, dtype=np.int64), np.empty(0, dtype=np.int64)
    counts = _count_pairs(pairs, N)
    row_ptr = np.zeros(N + 1, dtype=np.int64)
    np.cumsum(counts, out=row_ptr[1:])
    col_idx = _fill_csr_from_pairs(pairs, row_ptr)
    return row_ptr, col_idx


def compute_smoothing_lengths(pos, mass, h, eta=1.2,
                              niter=3, h_min=1e-4, h_max=0.5):
    """Iteratively update h and rho.

    Builds the CSR neighbour list ONCE using the current max(h).  Since
    h is clipped to [h_min, h_max], a single tree covers all possible
    kernel supports across iterations.  Returns (h, rho, row_ptr, col_idx).
    """
    N = len(pos)
    rho = np.ones(N) * 1e-5
    # Build CSR once — any h increase during iteration stays within the
    # tree radius because the density kernel returns 0 beyond 2h.
    radius = 2.0 * float(np.max(h))
    row_ptr, col_idx = build_neighbour_csr(pos, radius)
    for _ in range(niter):
        rho = sph_density_neigh(pos, mass, h, row_ptr, col_idx)
        h_new = eta * np.sqrt(mass / rho)
        h = np.clip(h_new, h_min, h_max)
    return h, rho, row_ptr, col_idx


def compute_timestep(h, cs, acc_sph, divv, p):
    """Adaptive global timestep."""
    absa = np.sqrt(acc_sph[:, 0] ** 2 + acc_sph[:, 1] ** 2)
    adivv = np.abs(divv)
    alpha = p["alpha_visc"]
    beta = p["beta_visc"]
    eps = p["eps_acc"]
    gcfl = p["gamma_CFL"]

    dt1 = h / (h * adivv + cs)
    dt2 = np.sqrt(h / (absa + eps))
    dt3 = h / ((1.0 + 1.2 * alpha) * cs + (1.0 + 1.2 * beta) * h * adivv)
    dt_i = gcfl * np.minimum(dt1, np.minimum(dt2, dt3))
    dt = float(np.min(dt_i))
    dt = max(dt, p["dt_min"])
    dt = min(dt, p["dt_max_tidal"])
    return dt


def inject_particle(t, p, c):
    """Return (r_new [1,2], v_new [1,2]) for a newly injected particle.

    Particles are placed at r_inj (user-set) with a small radial scatter
    (±20%).  If v_inj == 0 the tangential speed defaults to the local
    Keplerian velocity; otherwise the user value is used.  A thermal
    dispersion (~ CS) is added in both cases.
    """
    phase = np.random.uniform(0, 2 * np.pi)
    cos_p, sin_p = np.cos(phase), np.sin(phase)

    r_inj = p["r_inj"] * (1.0 + 0.2 * (2.0 * np.random.rand() - 1.0))
    r_new = r_inj * np.array([[cos_p, sin_p]])

    # Tangential speed: user value or local Keplerian
    if p["v_inj"] > 0.0:
        v_tan_base = p["v_inj"]
    else:
        v_tan_base = np.sqrt(c["GM_tot"] / r_inj)
    v_tan = v_tan_base + p["CS"] * np.random.randn()
    v_rad = p["CS"] * np.random.randn()

    v_new = np.array([[
        -v_tan * sin_p + v_rad * cos_p,
         v_tan * cos_p + v_rad * sin_p
    ]])
    return r_new, v_new


# ======================================================================
#  MAIN SIMULATION LOOP
# ======================================================================
def run_simulation(p, c, logfile):
    """Run the full SPH simulation.  Returns (snapshots, ts_data)."""

    # Seed the ring with N_seed particles on near-circular Keplerian orbits
    N_seed = int(p.get("N_seed", 1000))
    if N_seed > 0:
        theta_seed = np.random.uniform(0, 2 * np.pi, N_seed)
        r_seed = p["r_inj"] * (1.0 + 0.15 * np.random.randn(N_seed))
        r_seed = np.clip(r_seed, p["r_remove_in"] + 0.01, p["r_remove_out"] - 0.1)
        pos = np.column_stack([r_seed * np.cos(theta_seed),
                               r_seed * np.sin(theta_seed)])
        v_circ_seed = np.sqrt(c["GM_tot"] / r_seed)
        v_therm = p["CS"] * np.random.randn(N_seed)
        v_tan = v_circ_seed + v_therm
        vel = np.column_stack([-v_tan * np.sin(theta_seed),
                                v_tan * np.cos(theta_seed)])
        mass = np.full(N_seed, p["m_sph"])
        h_init = p["eta"] * np.sqrt(p["m_sph"] / p["rho_ref"])
        h = np.full(N_seed, h_init)
        rho = np.full(N_seed, p["rho_ref"])
        print(f"  Seeded ring with {N_seed} particles at <r> ~ {p['r_inj']:.3f} au")
    else:
        pos = np.empty((0, 2))
        vel = np.empty((0, 2))
        mass = np.empty(0)
        h = np.empty(0)
        rho = np.empty(0)
    t = 0.0
    dt = 1e-4
    t_end = p["t_end"]

    snapshots = []      # (t, pos_copy, rho_copy, h_copy, r2_copy)
    ts_data = []        # (t, mean_r, std_r)
    t_next_snap = 0.0
    dt_snap = p["dt_snap"]

    step = 0
    wall_start = time.time()
    pbar = tqdm(total=t_end, unit="yr", desc="Simulating",
                bar_format="{l_bar}{bar}| {n:.2f}/{total:.1f} yr "
                           "[{elapsed}<{remaining}, {rate_fmt}]")

    # Pre-compile Numba kernels with tiny dummy arrays
    _dp = np.zeros((2, 2))
    _dv = np.zeros((2, 2))
    _dm = np.ones(2) * 1e-6
    _dh = np.ones(2) * 0.01
    _dr = np.ones(2) * 1e-5
    _dpr = np.ones(2) * 100.0
    _row = np.array([0, 1, 2], dtype=np.int64)
    _col = np.array([1, 0], dtype=np.int64)
    _pairs = np.array([[0, 1]], dtype=np.int64)
    print("  Compiling Numba kernels (first call)...", flush=True)
    _ = kernel_W(0.1, 0.1, 0.01)
    _ = kernel_and_grad(0.1, 0.1, 0.01)
    _ = sph_density_neigh(_dp, _dm, _dh, _row, _col)
    _ = sph_forces_neigh(_dp, _dv, _dm, _dr, _dpr, _dh,
                         1.0, 2.0, 1.056, _row, _col)
    _ = _count_pairs(_pairs, 2)
    _ = _fill_csr_from_pairs(_pairs, np.array([0, 1, 2], dtype=np.int64))
    print("  Numba compilation done.", flush=True)

    # Extract scalars from dicts once (avoid per-step dict lookups)
    GM_tot = c["GM_tot"]
    GM2 = c["GM2"]
    eps = p["eps_grav"]
    cs = p["CS"]
    cs2 = cs * cs
    alpha_v = p["alpha_visc"]
    beta_v = p["beta_visc"]
    N_dot = c["N_dot"]
    dt_max_t = c["dt_max_tidal"]
    p["dt_max_tidal"] = dt_max_t
    eta = p["eta"]
    h_iter = p["h_iter"]
    h_min = p["h_min"]
    h_max = p["h_max"]
    r_remove_in = p["r_remove_in"]
    r_remove_out = p["r_remove_out"]

    # ------------------------------------------------------------------
    #  Compute initial forces (reused as start-of-first-step forces)
    # ------------------------------------------------------------------
    N = len(pos)
    if N > 0:
        h, rho, row_ptr, col_idx = compute_smoothing_lengths(
            pos, mass, h, eta=eta, niter=h_iter, h_min=h_min, h_max=h_max)
        P_over_rho2 = cs2 / rho
        acc_sph, divv = sph_forces_neigh(
            pos, vel, mass, rho, P_over_rho2, h,
            alpha_v, beta_v, cs, row_ptr, col_idx)
        r2_pos = companion_position(t, c)
        acc = acc_sph + central_gravity(pos, GM_tot, eps)
        if np.hypot(r2_pos[0], r2_pos[1]) <= 2.0:
            acc += tidal_force(pos, r2_pos, GM2, eps)
    else:
        acc = np.empty((0, 2))
        divv = np.empty(0)
        acc_sph = np.empty((0, 2))

    # ------------------------------------------------------------------
    #  Main loop — velocity-Verlet (KDK) with force caching
    #
    #  Forces computed ONCE per step (after drift).  End-of-step forces
    #  are reused as start-of-next-step forces.
    # ------------------------------------------------------------------
    while t < t_end:
        N = len(pos)

        # --- Adaptive timestep from CURRENT (cached) forces ---
        if N > 0:
            dt = compute_timestep(h, cs, acc_sph, divv, p)
        else:
            dt = 1e-4

        # --- Half-kick (in-place) ---
        if N > 0:
            vel += 0.5 * dt * acc

        # --- Drift (in-place) ---
        if N > 0:
            pos += dt * vel
        t += dt

        # --- Inject ---
        if N_dot > 0 and np.random.rand() < N_dot * dt:
            r_new, v_new = inject_particle(t, p, c)
            if N == 0:
                pos = r_new
                vel = v_new
            else:
                pos = np.vstack([pos, r_new])
                vel = np.vstack([vel, v_new])
            mass = np.append(mass, p["m_sph"])
            h = np.append(h, c["h_new_default"])
            rho = np.append(rho, p["rho_ref"])
            N += 1

        # --- Remove ---
        if N > 0:
            r_mag = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            keep = (r_mag > r_remove_in) & (r_mag < r_remove_out)
            if not np.all(keep):
                pos = pos[keep]
                vel = vel[keep]
                mass = mass[keep]
                h = h[keep]
                rho = rho[keep]
                N = len(pos)

        # --- Compute forces at NEW position (single evaluation) ---
        r2_pos = companion_position(t, c)
        if N > 0:
            h, rho, row_ptr, col_idx = compute_smoothing_lengths(
                pos, mass, h, eta=eta, niter=h_iter, h_min=h_min, h_max=h_max)
            P_over_rho2 = cs2 / rho
            acc_sph, divv = sph_forces_neigh(
                pos, vel, mass, rho, P_over_rho2, h,
                alpha_v, beta_v, cs, row_ptr, col_idx)
            acc = acc_sph + central_gravity(pos, GM_tot, eps)
            if np.hypot(r2_pos[0], r2_pos[1]) <= 2.0:
                acc += tidal_force(pos, r2_pos, GM2, eps)
        else:
            acc = np.empty((0, 2))
            acc_sph = np.empty((0, 2))
            divv = np.empty(0)

        # --- Second half-kick (in-place) ---
        if N > 0:
            vel += 0.5 * dt * acc

        # --- Snapshot (denser near periastron) ---
        peri_near = abs(t - c["t_peri"]) < 1.5
        snap_dt = dt_snap * 0.05 if peri_near else dt_snap
        if t >= t_next_snap:
            snapshots.append((t, pos.copy(), rho.copy(), h.copy(), r2_pos.copy()))
            t_next_snap = t + snap_dt

        # --- Progress bar + diagnostics (compute r_mag once) ---
        if N > 0:
            rmag = np.sqrt(pos[:, 0] ** 2 + pos[:, 1] ** 2)
            mr = float(np.mean(rmag))
        else:
            mr = 0.0

        # Record time-series on snapshot steps
        if t >= t_next_snap - snap_dt and N > 0:
            sr = float(np.std(rmag))
            ts_data.append((t, mr, sr))

        pbar.update(dt)
        pbar.set_postfix_str(f"N={N:d}  dt={dt:.1e}  <r>={mr:.3f}")

        # Log every 0.5 yr
        if step == 0 or (t - dt < (int((t - dt) * 2) + 1) * 0.5 <= t):
            msg = (f"t={t:.2f} yr  N={N:d}  dt={dt:.2e} yr  "
                   f"<r>={mr:.3f} au  wall={time.time() - wall_start:.0f}s")
            logfile.write(msg + "\n")

        step += 1

    pbar.close()
    wall_total = time.time() - wall_start
    summary = f"Simulation complete: {step} steps, {wall_total:.1f}s wall time"
    print(f"\n  {summary}")
    logfile.write(f"\n{summary}\n")
    return snapshots, ts_data


# ======================================================================
#  OUTPUT: GAUSSIAN DENSITY RENDERING
# ======================================================================
def render_sph_density(pos, h, m_sph, grid_size, extent):
    """Render SPH particles as a smooth density image via Gaussian splatting.

    Each particle's mass is deposited into a 2D histogram, then smoothed
    with a Gaussian kernel whose width matches the mean SPH smoothing
    length.  The result is a surface-density image with a fluid-like
    appearance.
    """
    xmin, xmax, ymin, ymax = extent
    if len(pos) == 0:
        return np.zeros((grid_size, grid_size))

    H, _, _ = np.histogram2d(
        pos[:, 0], pos[:, 1], bins=grid_size,
        range=[[xmin, xmax], [ymin, ymax]],
        weights=np.full(len(pos), m_sph))

    dx = (xmax - xmin) / grid_size
    dy = (ymax - ymin) / grid_size
    sigma_pix = np.clip(np.mean(h) / dx, 1.0, 20.0)

    img = gaussian_filter(H.T, sigma=sigma_pix)
    pixel_area = dx * dy
    img /= pixel_area

    return img


def _select_frames(snapshots, n_frames, t_peri, peri_window=1.5,
                   peri_fraction=0.7):
    """Select frame indices with denser sampling around periastron."""
    total = len(snapshots)
    if total == 0:
        return []
    times = np.array([s[0] for s in snapshots])

    in_window = ((times >= t_peri - peri_window) &
                 (times <= t_peri + peri_window))
    idx_in = np.where(in_window)[0]
    idx_out = np.where(~in_window)[0]

    if len(idx_in) == 0 or len(idx_out) == 0:
        print(f"    No periastron snapshots found near t={t_peri:.2f} yr, "
              f"using uniform sampling.")
        return np.linspace(0, total - 1, min(n_frames, total),
                           dtype=int).tolist()

    n_frames_in = max(int(n_frames * peri_fraction), 1)
    n_frames_out = n_frames - n_frames_in

    sel_in = idx_in[np.linspace(0, len(idx_in) - 1,
                                min(n_frames_in, len(idx_in)),
                                dtype=int)]
    sel_out = idx_out[np.linspace(0, len(idx_out) - 1,
                                  min(n_frames_out, len(idx_out)),
                                  dtype=int)]
    indices = sorted(set(sel_in.tolist() + sel_out.tolist()))

    dt_peri = 2 * peri_window
    dt_rest = times[-1] - times[0] - dt_peri
    speed_peri = dt_peri / (len(sel_in) / 10.0) if len(sel_in) > 0 else 0
    speed_rest = dt_rest / (len(sel_out) / 10.0) if len(sel_out) > 0 else 0
    print(f"    Frame allocation: {len(sel_in)} periastron + "
          f"{len(sel_out)} normal = {len(indices)} total")
    print(f"    Playback speed:  periastron {speed_peri:.2f} yr/s  |  "
          f"rest {speed_rest:.2f} yr/s  "
          f"({speed_rest/speed_peri:.0f}x slowdown)" if speed_peri > 0
          else "")

    return indices


def _adaptive_linear_limits(values, pad_frac=0.06, min_pad=1e-3):
    """Return padded linear-axis limits for 1D data."""
    vmin = float(np.min(values))
    vmax = float(np.max(values))
    span = vmax - vmin
    pad = max(span * pad_frac, min_pad, abs(vmax) * 1e-4)
    if span == 0.0:
        pad = max(pad, abs(vmax) * pad_frac, min_pad)
    return vmin - pad, vmax + pad


def make_mp4(snapshots, filename, m_sph, n_frames=300, fps=10,
             t_peri=11.455, plot_lim=1.5, render_grid=256):
    """Produce an MP4 with smooth Gaussian-splatted fluid rendering.

    Each particle is rendered as a Gaussian blob, producing a smooth
    density field that looks like a liquid.  Frames are sampled more
    densely around periastron.
    """
    total = len(snapshots)
    if total == 0:
        print("  No snapshots to animate.")
        return

    indices = _select_frames(snapshots, n_frames, t_peri)
    frames = [snapshots[i] for i in indices]
    companion_visible = [float(np.hypot(r2_f[0], r2_f[1])) < 3.0
                         for _, _, _, _, r2_f in frames]
    lim = max(float(plot_lim), 0.05)
    extent = [-lim, lim, -lim, lim]

    fig, ax = plt.subplots(figsize=(7, 7))
    fig.patch.set_facecolor("k")
    ax.set_facecolor("k")
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("x [au]", color="w")
    ax.set_ylabel("y [au]", color="w")
    ax.tick_params(colors="w")
    for spine in ax.spines.values():
        spine.set_edgecolor("w")

    # Reference circles
    theta_c = np.linspace(0, 2 * np.pi, 300)
    ax.plot(0.15 * np.cos(theta_c), 0.15 * np.sin(theta_c),
            "--", color="white", alpha=0.4, lw=0.8)
    ax.plot(0.30 * np.cos(theta_c), 0.30 * np.sin(theta_c),
            "--", color="white", alpha=0.4, lw=0.8)

    # Star marker
    ax.plot(0, 0, "*", color="white", ms=15, zorder=5)

    # Initial density image
    t_f, pos_f, rho_f, h_f, r2_f = frames[0]
    img0 = render_sph_density(pos_f, h_f, m_sph, render_grid, extent)
    density_norm = LogNorm(vmin=1e-5, vmax=1e-2, clip=True)
    im = ax.imshow(img0, extent=extent, origin='lower',
                   cmap='inferno', norm=density_norm,
                   aspect='equal', interpolation='bilinear', zorder=1)

    comp_marker, = ax.plot([], [], "o", color="white", ms=7, zorder=5)
    comp_trail = LineCollection([], linewidths=2.2, zorder=4, capstyle="round")
    ax.add_collection(comp_trail)
    title = ax.set_title("", color="white", fontsize=12)

    # Colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.02, shrink=0.8)
    cbar.set_label(r"Surface density [$M_\odot$ au$^{-2}$]", color="w")
    cbar.ax.yaxis.set_tick_params(color="w")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="w")

    def update(frame_idx):
        t_f, pos_f, rho_f, h_f, r2_f = frames[frame_idx]

        img = render_sph_density(pos_f, h_f, m_sph, render_grid, extent)
        im.set_data(img)

        if companion_visible[frame_idx]:
            comp_marker.set_data([r2_f[0]], [r2_f[1]])
            trail_start = max(0, frame_idx - 23)
            trail_points = []
            for idx in range(frame_idx, trail_start - 1, -1):
                if not companion_visible[idx]:
                    break
                trail_points.append(frames[idx][4][:2])
            trail_points.reverse()
            if len(trail_points) >= 2:
                pts = np.asarray(trail_points, dtype=float)
                segments = np.stack([pts[:-1], pts[1:]], axis=1)
                colors = np.ones((len(segments), 4), dtype=float)
                colors[:, 3] = np.linspace(0.05, 0.85, len(segments))
                comp_trail.set_segments(segments)
                comp_trail.set_color(colors)
            else:
                comp_trail.set_segments([])
        else:
            comp_marker.set_data([], [])
            comp_trail.set_segments([])
        title.set_text(
            f"SPH Simulation  |  Elapsed Time: {t_f:.2f} years")
        return im, comp_marker, comp_trail, title

    ani = animation.FuncAnimation(fig, update, frames=len(frames), blit=False)

    print(f"    Writing {len(frames)} frames...", flush=True)
    try:
        writer = animation.FFMpegWriter(fps=fps, bitrate=3000)
        ani.save(filename, writer=writer)
        print(f"  Animation saved: {filename}")
    except Exception as e:
        print(f"  FFMpeg failed ({e}), falling back to GIF...")
        gif_name = filename.replace(".mp4", ".gif")
        writer = animation.PillowWriter(fps=fps)
        ani.save(gif_name, writer=writer)
        print(f"  Animation saved: {gif_name}")
    plt.close(fig)


# ======================================================================
#  OUTPUT: TIME-SERIES PANEL
# ======================================================================
def make_time_series(ts_data, filename, t_peri=11.455):
    """Mean-radius vs time diagnostic plot."""
    if not ts_data:
        print("  No time-series data.")
        return
    ts = np.array(ts_data)
    t_arr = ts[:, 0]
    mr = ts[:, 1]
    sr = ts[:, 2]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_arr, mr, "b-", lw=1.2, label=r"$\langle r \rangle$")
    ax.fill_between(t_arr, mr - sr, mr + sr, alpha=0.3, color="blue")
    ax.axhline(0.15, ls="--", color="gray", lw=0.8, label="r = 0.15 au")
    ax.axhline(0.30, ls="--", color="orange", lw=0.8, label="r = 0.30 au")
    ax.axvline(t_peri, ls="--", color="red", lw=0.8, label="periastron")
    x_lo, x_hi = _adaptive_linear_limits(t_arr, pad_frac=0.03, min_pad=0.1)
    y_vals = np.concatenate([mr - sr, mr + sr, np.array([0.15, 0.30])])
    y_lo, y_hi = _adaptive_linear_limits(y_vals, pad_frac=0.08, min_pad=0.01)
    ax.set_xlim(max(0.0, x_lo), x_hi)
    ax.set_ylim(max(0.0, y_lo), y_hi)
    ax.set_xlabel("Time [yr]")
    ax.set_ylabel(r"$\langle r \rangle$ [au]")
    ax.set_title("Ring — Mean Radius vs Time")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(filename, dpi=150)
    print(f"  Time-series plot saved: {filename}")
    plt.close(fig)


# ======================================================================
#  MAIN
# ======================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("  2D SPH Binary Simulation")
    print("=" * 70)

    # Copy defaults
    params = dict(DEFAULTS)

    # Interactive parameter panel
    params = parameter_panel(params)
    active_numba_threads = configure_numba_threads(params)

    # Derive constants
    c = derive_constants(params)
    # Package constants dict for helpers
    c["e2_val"] = params["e2"]
    c["a2_val"] = params["a2"]

    # Print derived quantities
    print("\n  --- Derived quantities ---")
    print(f"  GM_tot       = {c['GM_tot']:.4f} au^3 yr^-2")
    print(f"  GM2          = {c['GM2']:.4f} au^3 yr^-2")
    print(f"  P2           = {c['P2']:.4f} yr")
    print(f"  t_peri       = {c['t_peri']:.4f} yr")
    print(f"  N_dot        = {c['N_dot']:.2f} yr^-1")
    print(f"  dt_max_tidal = {c['dt_max_tidal']:.2e} yr")
    print(f"  Numba threads= {active_numba_threads:d}")

    # Create timestamped output folder
    basedir = os.path.dirname(os.path.abspath(__file__))
    run_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(basedir, f"run_{run_stamp}")
    os.makedirs(outdir, exist_ok=True)
    print(f"\n  Output folder: {outdir}")

    # Open diagnostics log
    logpath = os.path.join(outdir, "diagnostics.log")
    with open(logpath, "w") as logfile:
        logfile.write(f"SPH Simulation — Diagnostics Log\n")
        logfile.write(f"Run started: {run_stamp}\n\n")
        logfile.write("=== User Parameters ===\n")
        for k, v in params.items():
            logfile.write(f"  {k:20s} = {v}\n")
        logfile.write(f"  {'active_numba_threads':20s} = {active_numba_threads}\n")
        logfile.write("\n=== Derived Constants ===\n")
        for k, v in c.items():
            logfile.write(f"  {k:20s} = {v}\n")
        logfile.write("\n=== Simulation Progress ===\n")
        logfile.flush()

        print("  Starting simulation...\n")
        snapshots, ts_data = run_simulation(params, c, logfile)

        logfile.write(f"\nTotal snapshots: {len(snapshots)}\n")
        logfile.write(f"Total ts_data points: {len(ts_data)}\n")

    # Outputs
    mp4_path = os.path.join(outdir, "sph_simulation.mp4")
    ts_path = os.path.join(outdir, "time_series.png")

    print("\n  Generating animation...")
    make_mp4(snapshots, mp4_path, m_sph=params["m_sph"],
             n_frames=params["n_frames"], fps=params["fps"],
             t_peri=c["t_peri"], plot_lim=params["plot_lim"],
             render_grid=int(params["render_grid"]))

    print("  Generating time-series plot...")
    make_time_series(ts_data, ts_path, t_peri=c["t_peri"])

    print(f"\n  All outputs in: {outdir}")
    print("  Done!\n")
