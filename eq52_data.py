import numpy as np
import torch
from utils import unpack

# --Full viscosity and diffusivity--
v_z = 0.01
v_h = 0.01
k_z = 0.01
k_h = 0.01


def Q(xzt):
    x, z, t = unpack(xzt)
    return np.pi * torch.cos(2 * np.pi * x) * torch.sin(4 * np.pi * z) * torch.exp(-4 * np.pi**2 * (v_h + v_z + k_z) * t)


def benchmark_solution(xzt):
    x, z, t = unpack(xzt)
    u = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    p = 0.25 * np.cos(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (v_h + v_z) * t) + np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * k_z * t) / (2 * np.pi)
    T = np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * k_z * t)

    return np.hstack((u, w, p, T))


def benchmark_du(xzt):
    x, z, t = unpack(xzt)
    exp_factor = np.exp(-4 * np.pi**2 * (v_h + v_z) * t)
    du_x = -2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z) * exp_factor
    du_z = 2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z) * exp_factor
    return [du_x, du_z]


def benchmark_dw(xzt):
    x, z, t = unpack(xzt)
    exp_factor = np.exp(-4 * np.pi ** 2 * (v_h + v_z) * t)
    dw_x = -2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z) * exp_factor
    dw_z = 2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z) * exp_factor
    return [dw_x, dw_z]


def benchmark_dp(xzt):
    x, z, t = unpack(xzt)
    dp_x = -np.pi * np.sin(4 * np.pi * x) * np.exp(-8 * np.pi**2 * (v_h + v_z) * t)
    dp_z = -np.sin(2 * np.pi * z) * np.exp(-4 * np.pi**2 * k_z * t)

    return [dp_x, dp_z]


def benchmark_dT(xzt):
    x, z, t = unpack(xzt)
    dT_x = 0.0 * x
    dT_z = 2 * np.pi * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi**2 * k_z * t)
    return [dT_x, dT_z]


def init_cond_u(xzt):
    """
    Initial condition for u.
    """
    x, z, t = unpack(xzt)
    return -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z)


def init_cond_du_x(xzt):
    x, z, t = unpack(xzt)
    return -2 * np.pi * np.cos(2 * np.pi * x) * np.cos(2 * np.pi * z)


def init_cond_du_z(xzt):
    x, z, t = unpack(xzt)
    return 2 * np.pi * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * z)


def init_cond_T(xzt):
    """
    Initial condition for T.
    """
    x, z, t = unpack(xzt)
    return np.sin(2 * np.pi * z)


def init_cond_dT_x(xzt):
    x, z, t = unpack(xzt)
    return 0.0 * x


def init_cond_dT_z(xzt):
    x, z, t = unpack(xzt)
    return 2 * np.pi * np.cos(2 * np.pi * z)
