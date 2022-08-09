import deepxde as dde
import numpy as np
import torch
from utils import get_params, plot_all_output3d, plot_error2d
import os
from joblib import Parallel, delayed
import time

# --Setup space and time domains--
x_min, x_max = 0.0, 1.0
z_min, z_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0


def get_geomtime():
    space_domain = dde.geometry.Rectangle([x_min, z_min], [x_max, z_max])
    time_domain = dde.geometry.TimeDomain(t_min, t_max)
    return dde.geometry.GeometryXTime(space_domain, time_domain)


def primitive_residual_l2(xzt, uwpT, Q, v_z, v_h, k_z, k_h):
    """
    PDE L2 interior residuals.
    """
    u = uwpT[:, 0:1]
    w = uwpT[:, 1:2]
    p = uwpT[:, 2:3]
    T = uwpT[:, 3:4]
    du_x = dde.grad.jacobian(uwpT, xzt, i=0, j=0)
    du_z = dde.grad.jacobian(uwpT, xzt, i=0, j=1)
    du_t = dde.grad.jacobian(uwpT, xzt, i=0, j=2)
    dw_z = dde.grad.jacobian(uwpT, xzt, i=1, j=1)
    dp_x = dde.grad.jacobian(uwpT, xzt, i=2, j=0)
    dp_z = dde.grad.jacobian(uwpT, xzt, i=2, j=1)
    dT_x = dde.grad.jacobian(uwpT, xzt, i=3, j=0)
    dT_z = dde.grad.jacobian(uwpT, xzt, i=3, j=1)
    dT_t = dde.grad.jacobian(uwpT, xzt, i=3, j=2)
    du_xx = dde.grad.hessian(uwpT, xzt, component=0, i=0, j=0)
    du_zz = dde.grad.hessian(uwpT, xzt, component=0, i=1, j=1)
    dT_xx = dde.grad.hessian(uwpT, xzt, component=3, i=0, j=0)
    dT_zz = dde.grad.hessian(uwpT, xzt, component=3, i=1, j=1)
    with torch.no_grad():
        q = Q(xzt)

    # PDE residuals
    pde1 = du_t + u * du_x + w * du_z - v_h * du_xx - v_z * du_zz + dp_x
    pde2 = dp_z + T
    pde3 = du_x + dw_z
    pde4 = dT_t + u * dT_x + w * dT_z - k_h * dT_xx - k_z * dT_zz - q

    return [pde1, pde2, pde3, pde4]


# TODO: Implement H1 norm residuals
def primitive_residual_h1(xzt, uwpT, Q, v_z, v_h, k_z, k_h):
    """
    PDE H1 interior residuals.
    """
    pass


def initial_conditions(init_cond_u, init_cond_T):
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
def learn_primitive_equations(equation, outdir):
    # Get output directory name
    match equation:
        case '5.2':
            import eq52_data as eq_data
        case '5.3':
            import eq53_data as eq_data
        case _:
            raise ValueError(f'Eq {equation} is not a valid equation to solve.')

    # Necessary because numpy defaults to `float64`
    dde.config.set_default_float('float64')

    # Setup boundary and initial conditions
    geomtime = get_geomtime()
    ics = initial_conditions(eq_data.init_cond_u, eq_data.init_cond_T)
    bcs = boundary_conditions()

    data = dde.data.TimePDE(
        geomtime,
        lambda xzt, uwpT: primitive_residual_l2(xzt, uwpT, eq_data.Q, eq_data.v_z, eq_data.v_h, eq_data.k_z, eq_data.k_h),
        [*ics, *bcs],
        num_domain=5000,
        num_boundary=500,
        num_initial=1000,
        train_distribution='uniform',
        solution=eq_data.benchmark_solution
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
    loss_history, train_state = model.train(iterations=int(45000), display_every=1000, model_save_path=f'{outdir}/model')
    dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir=outdir)

    times = np.array([0.0, 0.5, 1.0])
    plot_all_output3d(times, model.predict, equation, 50, outdir)
    plot_error2d(times, model.predict, eq_data.benchmark_solution, eq_data.benchmark_dp, equation, 50, outdir)


if __name__ == '__main__':
    n_jobs = 2
    start = time.time()
    print(f'Job initiated at time {start}.')
    Parallel(n_jobs=n_jobs)(delayed(learn_primitive_equations)(eq, f'run{eq}') for eq in ['5.2', '5.3'])
    end = time.time()
    print(f'{n_jobs} finished running in {end - start} seconds.')
