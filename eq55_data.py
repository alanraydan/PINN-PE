import numpy as np
from utils import unpack

# --Only horizontal viscosity and diffusivity--
v_z = 0.0
v_h = 0.01
k_z = 0.0
k_h = 0.01


def Q(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x


def benchmark_solution(xzt):
    x, z, t = unpack(xzt)
    u = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * v_h * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * v_h * t)
    p = 0.25 * np.cos(4 * np.pi * x) * np.exp(-8 * np.pi**2 * v_h * t)
    T = 0.0 * x
    return np.hstack((u, w, p, T))


def benchmark_du(xzt):
    x, z, t = unpack(xzt)
    exp_factor = np.exp(-4 * np.pi**2 * v_h * t)
    du_x = -2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z) * exp_factor
    du_z = 2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z) * exp_factor
    return [du_x, du_z]


def benchmark_dw(xzt):
    x, z, t = unpack(xzt)
    exp_factor = np.exp(-4 * np.pi ** 2 * v_h * t)
    dw_x = -2 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * exp_factor
    dw_z = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z) * exp_factor
    return [dw_x, dw_z]


def benchmark_dp(xzt):
    x, z, t = unpack(xzt)
    dp_x = -np.pi * np.sin(4 * np.pi * x) * np.exp(-8 * np.pi**2 * v_h * t)
    dp_z = 0.0 * x
    return [dp_x, dp_z]


def benchmark_dT(xzt):
    x, z, t = unpack(xzt)
    dT_x = 0.0 * x
    dT_z = 0.0 * x
    return [dT_x, dT_z]


def init_cond_u(xzt):
    x, z, t = unpack(xzt)
    return -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z)


def init_cond_du_x(xzt):
    x, z, t = unpack(xzt)
    return -2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z)


def init_cond_du_z(xzt):
    x, z, t = unpack(xzt)
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z)


def init_cond_T(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x


def init_cond_dT_x(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x


def init_cond_dT_z(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x
