import deepxde as dde
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

# --Full viscosity and diffusivity--
v_z = 1.0
v_h = 1.0
k_z = 1.0
k_h = 1.0
Q = lambda x, z, t: 0.0 * x

# --Setup space and time domains--
x_min, x_max = 0.0, 1.0
z_min, z_max = 0.0, 1.0
t_min, t_max = 0.0, 1.0
space_domain = dde.geometry.Rectangle([x_min, z_min], [x_max, z_max])
time_domain = dde.geometry.TimeDomain(t_min, t_max)
geomtime = dde.geometry.GeometryXTime(space_domain, time_domain)


# --Reference solution--
# This is technically incorrect; waiting for updated solution
def solution(xzt):
    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    t = xzt[:, 2:3]

    u = -np.sin(np.pi * x) * np.cos(np.pi * z) * np.exp(-np.pi * np.pi * (v_h + v_z) * t)
    w = np.cos(np.pi * x) * np.sin(np.pi * z) * np.exp(-np.pi * np.pi * (v_h + v_z) * t)
    p = np.cos(2 * np.pi * x) * np.exp(-2 * np.pi * np.pi * (v_h + v_z) * t) / 4.0
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

    # PDE residuals
    pde1 = du_t + u * du_x + w * du_z - v_h * du_xx - v_z * du_zz + dp_x
    pde2 = dp_z + T
    pde3 = du_x + dw_z
    pde4 = dT_t + u * dT_x + w * dT_z - k_h * dT_xx - k_z * dT_zz

    return [pde1, pde2, pde3, pde4]


# --Initial conditions--
def init_cond_u(x):
    return np.sin(np.pi * x[:, 0:1]) * np.cos(np.pi * x[:, 1:2])


def init_cond_T(x):
    return 0.0 * x[:, 0:1]


# --Boundary conditions for each function--
def z_boundary(x, on_boundary):
    return on_boundary and (np.isclose(x[1], 0) or np.isclose(x[1], 1))


def plot_all_output3d(times, func, points_per_dim=25, filename=None):
    assert func.__name__ == 'predict' or func.__name__ == 'solution'
    prim_names = ('u', 'w', 'p', 'T')
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)
    # --Reshape arrays to match fun input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))
    fig, ax = plt.subplots(nrows=4, ncols=len(times), figsize=(13, 13), subplot_kw=dict(projection='3d'))
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
                ax[j, i].text(0.5, 0.5, 1, f'{prim_names[j]}', transform=ax[j, i].transAxes)
            if j == 0:
                ax[j, i].set_title(f't = {time}', loc='center', y=0.99)
    if filename is not None:
        fig.savefig(f'plots/{filename}')
    else:
        plt.show()


if __name__ == '__main__':
    # Numpy arrays default to float64 so this line is necessary
    dde.config.set_default_float('float64')

    ic_u = dde.icbc.IC(geomtime, init_cond_u, lambda _, on_initial: on_initial, component=0)
    ic_T = dde.icbc.IC(geomtime, init_cond_T, lambda _, on_initial: on_initial, component=3)
    ics = [ic_u, ic_T]
    bc_u_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_u_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=0)
    bc_du_z = dde.icbc.NeumannBC(geomtime, lambda x: 0, z_boundary, component=0)

    bc_w_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=1)
    bc_w_z = dde.icbc.DirichletBC(geomtime, lambda x: 0, z_boundary, component=1)

    bc_p_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_p_z = dde.icbc.PeriodicBC(geomtime, 1, lambda _, on_boundary: on_boundary, derivative_order=0, component=2)
    bc_dp_z = dde.icbc.NeumannBC(geomtime, lambda x: 0, z_boundary, component=2)

    bc_T_x = dde.icbc.PeriodicBC(geomtime, 0, lambda _, on_boundary: on_boundary, derivative_order=0, component=3)
    bc_T_z = dde.icbc.DirichletBC(geomtime, lambda x: 0, z_boundary, component=3)

    bcs = [bc_u_x, bc_u_z, bc_du_z, bc_w_x, bc_w_z, bc_p_x, bc_p_z, bc_dp_z, bc_T_x, bc_T_z]

    data = dde.data.TimePDE(
        geomtime,
        primitive_equations,
        [*ics, *bcs],
        num_domain=5000,
        num_boundary=500,
        num_initial=500,
        train_distribution='uniform',
        # solution=solution
    )

    # --Network architecture--
    layer_size = [3] + [128] * 2 + [4]
    activation = 'tanh'
    initializer = 'Glorot normal'
    net = dde.nn.FNN(layer_size, activation, initializer)

    model = dde.Model(data, net)
    model.compile('adam', lr=1e-4, loss='MSE')
    loss_history, train_state = model.train(epochs=2, display_every=1)
    # dde.saveplot(loss_history, train_state, issave=True, isplot=True)

    times = np.array([0.0, 0.5, 1.0])
    plot_all_output3d(times, model.predict, 50, 'test_plot')


