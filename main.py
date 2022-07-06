import deepxde as dde
import matplotlib.pyplot as plt
import numpy as np

# --Full viscosity and diffusivity--
v_z = 0.1
v_h = 0.1
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


# --Reference solution--
def benchmark_solution(xzt):
    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    t = xzt[:, 2:3]

    u = -np.sin(2 * np.pi * x) * np.cos(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    w = np.cos(2 * np.pi * x) * np.sin(2 * np.pi * z) * np.exp(-4 * np.pi * np.pi * (v_h + v_z) * t)
    p = np.cos(4 * np.pi * x) * np.exp(-8 * np.pi * np.pi * (v_h + v_z) * t) / 4.0
    T = 0.0 * x

    return np.hstack((u, w, p, T))


# --PDE interior residuals--
def primitive_equations(x, y):
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


# --Initial condition for u--
def init_cond_u(x):
    return -np.sin(2 * np.pi * x[:, 0:1]) * np.cos(2 * np.pi * x[:, 1:2])


# --Initial condition for T--
def init_cond_T(x):
    return 0.0 * x[:, 0:1]


def initial_conditions():
    geomtime = get_geomtime()
    ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
    ic_T = dde.icbc.IC(geomtime, init_cond_T, lambda _, on_initial: on_initial, component=3)
    return [ic_u, ic_T]


# --Boundary values for z component--
def z_boundary(x, on_boundary):
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


def plot_all_output3d(times, func, points_per_dim=25, filename=None):
    prim_names = ('u', 'w', 'p', 'T')
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)
    # --Reshape arrays to match func input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))
    fig, ax = plt.subplots(nrows=4, ncols=len(times), figsize=(13, 13), subplot_kw=dict(projection='3d'))
    title = func.__name__
    if func.__name__ == 'predict':
        title = 'PINN Output'
    fig.suptitle(title)
    fig.tight_layout()
    for i, time in enumerate(times):
        t = time * np.ones_like(x)
        xzt = np.hstack((x, z, t))
        out = func(xzt)
        for j in range(out.shape[1]):
            Out = out[:, j].reshape(X.shape)
            ax[j, i].plot_surface(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            if i == 0:
                # TODO: Find a way to format this more nicely
                ax[j, i].text(0.5, 0.5, 1, f'{prim_names[j]}', transform=ax[j, i].transAxes, fontsize='xx-large')
            if j == 0:
                ax[j, i].set_title(f't = {time}', y=0.99, fontsize='xx-large')
    if filename is not None:
        fig.savefig(f'plots/multi_nn/{filename}2')
    else:
        plt.show()


def plot_all_error2d(times, func, points_per_dim=25, filename=None):
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)
    # --Reshape arrays to match func input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))
    fig, ax = plt.subplots(nrows=4, ncols=len(times), figsize=(13, 13))
    title = f'{func.__name__} error'
    if func.__name__ == 'predict':
        title = 'PINN Error'
    fig.tight_layout()
    fig.suptitle(title)
    for i, time in enumerate(times):
        t = time * np.ones_like(x)
        xzt = np.hstack((x, z, t))
        out = np.abs(func(xzt) - benchmark_solution(xzt))
        for j in range(out.shape[1]):
            Out = out[:, j].reshape(X.shape)
            cs = ax[j, i].contourf(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            fig.colorbar(cs, ax=ax[j, i])
    if filename is not None:
        fig.savefig(f'plots/multi_nn/{filename}2')
    else:
        plt.show()


# --PINN setup and learning iterations--
def learn_primitive_equations():
    # Necessary because numpy defaults to `float64`
    dde.config.set_default_float('float64')

    geomtime = get_geomtime()
    ics = initial_conditions()
    bcs = boundary_conditions()

    data = dde.data.TimePDE(
        geomtime,
        primitive_equations,
        [*ics, *bcs],
        num_domain=5000,
        num_boundary=500,
        num_initial=500,
        train_distribution='uniform',
        solution=benchmark_solution
    )

    # --Network architecture--
    layer_size = [3, [64, 64, 64, 64], [64, 64, 64, 64], 4]
    activation = 'tanh'
    initializer = 'Glorot normal'
    # The PFNN class allows us to use 4 distinct NNs instead of just 1 with 4 outputs
    net = dde.nn.PFNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile('adam', lr=1e-4, loss='MSE')
    loss_history, train_state = model.train(iterations=int(3e4), display_every=1000)
    dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    times = np.array([0.0, 0.5, 1.0])
    plot_all_output3d(times, model.predict, 50, 'learned_model')
    plot_all_error2d(times, model.predict, 50, 'model_error')


if __name__ == '__main__':
    learn_primitive_equations()