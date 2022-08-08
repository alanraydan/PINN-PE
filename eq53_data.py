import numpy as np
from utils import unpack

# --Full viscosity and diffusivity--
v_z = 0.01
v_h = 0.01
k_z = 1.0
k_h = 1.0


def Q(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x


def benchmark_solution(xzt):
    x, z, t = unpack(xzt)
    u = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    p = np.cos(4 * np.pi * x) * np.exp(-8 * np.pi * np.pi * (v_h + v_z) * t) / 4.0
    T = 0.0 * x

    return np.hstack((u, w, p, T))

def benchmark_dp(xzt):
    x, z, t = unpack(xzt)
    dp_x = -np.pi * np.sin(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (v_h + v_z) * t)
    dp_z = np.zeros_like(x)

    return np.hstack((dp_x, dp_z))


def init_cond_u(xzt):
    """
    Initial condition for u.
    """
    x, z, t = unpack(xzt)
    return -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z)


def init_cond_T(xzt):
    """
    Initial condition for T.
    """
    x, z, t = unpack(xzt)
    return np.zeros_like(x)