import deepxde as dde
import numpy as np
from utils import get_params, plot_all_output3d, plot_error2d
import os

# --Full viscosity and diffusivity--
v_z = 0.01
v_h = 0.01
k_z = 1.0
k_h = 1.0
Q = lambda x, z, t: 0.0 * x

# --Setup space and time domains--
x_min, x_max = 0.0, 1.0
z_min, z_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0


def get_geomtime():
    space_domain = dde.geometry.Rectangle([x_min, z_min], [x_max, z_max])
    time_domain = dde.geometry.TimeDomain(t_min, t_max)
    return dde.geometry.GeometryXTime(space_domain, time_domain)


# TODO: benchmark values for dp_x, dp_z
def benchmark_solution(xzt):
    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    t = xzt[:, 2:3]

    u = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    p = np.cos(4 * np.pi * x) * np.exp(-8 * np.pi * np.pi * (v_h + v_z) * t) / 4.0
    T = 0.0 * x

    return np.hstack((u, w, p, T))


def primitive_residual_l2(x, y):
    """
    PDE L2 interior residuals.
    """
    u = y[:, 0:1]
    w = y[:, 1:2]
    p = y[:, 2:3]
    T = y[:, 3:4]
    du_x = dde.grad.jacobian(y, x, i=0, j=0)
    du_z = dde.grad.jacobian(y, x, i=0, j=1)
    du_t = dde.grad.jacobian(y, x, i=0, j=2)
    dw_z = dde.grad.jacobian(y, x, i=1, j=1)
    dp_x = dde.grad.jacobian(y, x, i=2, j=0)
    dp_z = dde.grad.jacobian(y, x, i=2, j=1)
    dT_x = dde.grad.jacobian(y, x, i=3, j=0)
    dT_z = dde.grad.jacobian(y, x, i=3, j=1)
    dT_t = dde.grad.jacobian(y, x, i=3, j=2)
    du_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    du_zz = dde.grad.hessian(y, x, component=0, i=1, j=1)
    dT_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    dT_zz = dde.grad.hessian(y, x, component=3, i=1, j=1)
    q = Q(x[:, 0:1], x[:, 1:2], x[:, 2:3])

    # PDE residuals
    pde1 = du_t + u * du_x + w * du_z - v_h * du_xx - v_z * du_zz + dp_x
    pde2 = dp_z + T
    pde3 = du_x + dw_z
    pde4 = dT_t + u * dT_x + w * dT_z - k_h * dT_xx - k_z * dT_zz - q

    return [pde1, pde2, pde3, pde4]


def primitive_residual_h1(x, y):
    """
    PDE H1 interior residuals.
    """


def init_cond_u(x):
    """
    Initial condition for u.
    """
    return -np.sin(2 * np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])


def init_cond_T(x):
    """
    Initial condition for T.
    """
    return 0.0 * x[:, 0:1]


def initial_conditions():
    geomtime = get_geomtime()
    ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
    ic_T = dde.icbc.IC(geomtime, init_cond_T, lambda _, on_initial: on_initial, component=3)
    return [ic_u, ic_T]


def z_boundary(x, on_boundary):
    """
    Boundary values for z component.
    """
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


def boundary_conditions():
    geomtime = get_geomtime()
    bc_u_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_u_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_w_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_w_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_w_z_Dirichlet = dde.icbc.DirichletBC(geomtime, lambda x: 0, z_boundary, component=1)
    bc_p_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_p_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_T_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
    bc_T_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
    return [bc_u_x, bc_u_z, bc_w_x, bc_w_z, bc_w_z_Dirichlet, bc_p_x, bc_p_z, bc_T_x, bc_T_z]


# --PINN setup and learning iterations--
def learn_primitive_equations():
    # Get output directory name
    outdir = get_params()

    # Necessary because numpy defaults to `float64`
    dde.config.set_default_float('float64')

    # Setup boundary and initial conditions
    geomtime = get_geomtime()
    ics = initial_conditions()
    bcs = boundary_conditions()

    data = dde.data.TimePDE(
        geomtime,
        primitive_residual_l2,
        [*ics, *bcs],
        num_domain=5000,
        num_boundary=500,
        num_initial=1000,
        train_distribution='uniform',
        solution=benchmark_solution
    )

    # Network architecture
    layer_size = [3, [32, 32, 32, 32], [32, 32, 32, 32], 4]
    activation = 'tanh'
    initializer = 'Glorot normal'
    # The PFNN class allows us to use 4 distinct NNs instead of just 1 with 4 outputs
    net = dde.nn.PFNN(layer_size, activation, initializer)

    # Instantiate and train model
    model = dde.Model(data, net)
    model.compile('adam', lr=1e-4, loss='MSE')
    if not os.path.exists(f'./{outdir}'):
        os.mkdir(f'./{outdir}')
    loss_history, train_state = model.train(iterations=int(42e3), display_every=1000, model_save_path=f'{outdir}/model')
    dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir=outdir)

    times = np.array([0.0, 0.5, 1.0])
    plot_all_output3d(times, model.predict, 50, outdir)
    plot_error2d(times, model.predict, benchmark_solution, 50, outdir)
    plot_error2d(times, model.predict, benchmark_solution, 50, outdir)


if __name__ == '__main__':
    learn_primitive_equations()
