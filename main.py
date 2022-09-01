import deepxde as dde
import numpy as np
from utils import plot_all_output3d, plot_error2d
import os
from joblib import Parallel, delayed
import sys

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
    q = Q(xzt)

    # PDE residuals
    pde1 = du_t + u * du_x + w * du_z - v_h * du_xx - v_z * du_zz + dp_x
    pde2 = dp_z + T
    pde3 = du_x + dw_z
    pde4 = dT_t + u * dT_x + w * dT_z - k_h * dT_xx - k_z * dT_zz - q

    return [pde1, pde2, pde3, pde4]


def primitive_residual_h1(xzt, uwpT, Q, v_z, v_h, k_z, k_h):
    """
    PDE H1 interior residuals.
    """
    pde1, pde2, pde3, pde4 = primitive_residual_l2(xzt, uwpT, Q, v_z, v_h, k_z, k_h)
    dpde1_x = dde.grad.jacobian(pde1, xzt, i=0, j=0)
    dpde1_z = dde.grad.jacobian(pde1, xzt, i=0, j=1)
    dpde2_x = dde.grad.jacobian(pde2, xzt, i=0, j=0)
    dpde2_z = dde.grad.jacobian(pde2, xzt, i=0, j=1)
    dpde3_x = dde.grad.jacobian(pde3, xzt, i=0, j=0)
    dpde3_z = dde.grad.jacobian(pde3, xzt, i=0, j=1)
    dpde4_x = dde.grad.jacobian(pde4, xzt, i=0, j=0)
    dpde4_z = dde.grad.jacobian(pde4, xzt, i=0, j=1)

    return [pde1, pde2, pde3, pde4, dpde1_x, dpde1_z, dpde2_x, dpde2_z, dpde3_x, dpde3_z, dpde4_x, dpde4_z]


def initial_conditions_l2(init_u, init_T):
    geomtime = get_geomtime()
    ic_u = dde.icbc.IC(geomtime, init_u, lambda _, on_initial: on_initial, component=0)
    ic_T = dde.icbc.IC(geomtime, init_T, lambda _, on_initial: on_initial, component=3)
    return [ic_u, ic_T]


def initial_conditions_h1(init_u, init_T, init_du_x, init_du_z, init_dT_x, init_dT_z):
    geomtime = get_geomtime()
    ic_u, ic_T = initial_conditions_l2(init_u, init_T)

    ic_du_x = dde.icbc.OperatorBC(geomtime,
                                  lambda xzt, uwpT, _: dde.grad.jacobian(uwpT, xzt, i=0, j=0) - init_du_x(xzt.detach()),
                                  lambda xzt, on_boundary: np.isclose(xzt[2], 0.0))
    ic_du_z = dde.icbc.OperatorBC(geomtime,
                                  lambda xzt, uwpT, _: dde.grad.jacobian(uwpT, xzt, i=0, j=1) - init_du_z(xzt.detach()),
                                  lambda xzt, on_boundary: np.isclose(xzt[2], 0.0))
    ic_dT_x = dde.icbc.OperatorBC(geomtime,
                                  lambda xzt, uwpT, _: dde.grad.jacobian(uwpT, xzt, i=3, j=0) - init_dT_x(xzt.detach()),
                                  lambda xzt, on_boundary: np.isclose(xzt[2], 0.0))
    ic_dT_z = dde.icbc.OperatorBC(geomtime,
                                  lambda xzt, uwpT, _: dde.grad.jacobian(uwpT, xzt, i=3, j=1) - init_dT_z(xzt.detach()),
                                  lambda xzt, on_boundary: np.isclose(xzt[2], 0.0))
    return [ic_u, ic_T, ic_du_x, ic_du_z, ic_dT_x, ic_dT_z]


def z_boundary(x, on_boundary):
    """
    Boundary values for z component.
    """
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


def boundary_conditions_l2():
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


def boundary_conditions_h1():
    geomtime = get_geomtime()
    bc_du_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)
    bc_du_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=1, component=0)
    bc_dw_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=1)
    bc_dw_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=1, component=1)
    bc_dw_z_Dirichlet = dde.icbc.OperatorBC(geomtime, lambda xzt, uwpT, _: dde.grad.jacobian(uwpT, xzt, i=1, j=0), z_boundary)
    bc_dp_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=2)
    bc_dp_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=1, component=2)
    bc_dT_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=1, component=3)
    bc_dT_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=1, component=3)
    return [*boundary_conditions_l2(), bc_du_x, bc_du_z, bc_dw_x, bc_dw_z, bc_dw_z_Dirichlet, bc_dp_x, bc_dp_z, bc_dT_x, bc_dT_z]


# --PINN setup and learning iterations--
def learn_primitive_equations(equation, iters, residual, outdir):
    assert residual == 'l2' or residual == 'h1'
    match equation:
        case '5.2':
            import eq52_data as eq_data
        case '5.3':
            import eq53_data as eq_data
        case '5.4':
            import eq54_data as eq_data
        case '5.5':
            import eq55_data as eq_data
        case _:
            raise ValueError(f'Eq {equation} is not a valid equation to solve.')

    # Necessary because numpy defaults to `float64`
    dde.config.set_default_float('float64')

    # Setup boundary and initial conditions
    geomtime = get_geomtime()

    if residual == 'h1':
        ics = initial_conditions_h1(
            eq_data.init_cond_u,
            eq_data.init_cond_T,
            eq_data.init_cond_du_x,
            eq_data.init_cond_du_z,
            eq_data.init_cond_dT_x,
            eq_data.init_cond_dT_z
        )
        bcs = boundary_conditions_h1()
        res = primitive_residual_h1
    if residual == 'l2':
        ics = initial_conditions_l2(eq_data.init_cond_u, eq_data.init_cond_T)
        bcs = boundary_conditions_l2()
        res = primitive_residual_l2

    data = dde.data.TimePDE(
        geomtime,
        lambda xzt, uwpT: res(xzt, uwpT, eq_data.Q, eq_data.v_z, eq_data.v_h, eq_data.k_z, eq_data.k_h),
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
    loss_history, train_state = model.train(iterations=int(iters), display_every=1000, model_save_path=f'{outdir}/model')
    dde.saveplot(loss_history, train_state, issave=True, isplot=True, output_dir=outdir)

    times = np.array([0.0, 0.5, 1.0])
    plot_all_output3d(times, model.predict, equation, 50, outdir)
    plot_error2d(times, model.predict, eq_data.benchmark_solution, eq_data.benchmark_dp, equation, 50, outdir)


if __name__ == '__main__':
    n_jobs = 2
    n_iters = 50_000
    eq = sys.argv[1]
    residuals = ['l2', 'h1']
    Parallel(n_jobs=n_jobs)(delayed(learn_primitive_equations)(eq, n_iters, res, f'eq{eq}_{n_iters}iters_{res}res') for res in residuals)

