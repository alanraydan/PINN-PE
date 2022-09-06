import deepxde as dde
import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_params():
    """
    Function for parsing parameters from config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('equation', help='Equation you would like to solve.')
    parser.add_argument('outdir', help='Directory to output data.')
    args = parser.parse_args()
    return args.equation, args.outdir


def unpack(xzt):
    x = xzt[:, 0:1]
    z = xzt[:, 1:2]
    t = xzt[:, 2:3]
    return x, z, t


def plot_all_output3d(times, func, equation, points_per_dim=25, outdir=None):
    prim_names = ('u', 'w', r'$\partial_x p$', r'$\partial_z p$', 'T')
    nrows = 5
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)

    # --Reshape arrays to match func input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))

    fig, ax = plt.subplots(nrows=nrows, ncols=len(times), figsize=(13, 16), subplot_kw=dict(projection='3d'))
    title = func.__name__
    if func.__name__ == 'predict':
        title = 'PINN Output'
    title += f' {equation}'
    fig.suptitle(title)

    for j in range(nrows):

        maximum = -np.inf
        minimum = np.inf

        for i, time in enumerate(times):
            t = time * np.ones_like(x)
            xzt = np.hstack((x, z, t))
            match j:
                case 0 | 1:
                    out = func(xzt)[:, j:j+1]
                case 2:
                    out = func(xzt, operator=lambda x, y: dde.grad.jacobian(y, x, i=2, j=0))
                case 3:
                    out = func(xzt, operator=lambda x, y: dde.grad.jacobian(y, x, i=2, j=1))
                case 4:
                    out = func(xzt)[:, j-1:j]
            maximum = max(out) if max(out) > maximum else maximum
            minimum = min(out) if min(out) < minimum else minimum
            Out = out.reshape(X.shape)
            ax[j, i].plot_surface(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            if i == 0:
                ax[j, i].text2D(-0.2, 0.5, f'{prim_names[j]}', transform=ax[j, i].transAxes, fontsize='x-large')
            if j == 0:
                ax[0, i].set_title(f't = {time}', y=0.99, fontsize='x-large')
        ax[j, 0].set_zlim(minimum, maximum)
        ax[j, 1].set_zlim(minimum, maximum)
        ax[j, 2].set_zlim(minimum, maximum)
    fig.subplots_adjust(0.02, 0.06, 0.95, 0.95)

    if outdir is not None:
        fig.savefig(f'{outdir}/model_output')
    else:
        plt.show()


def plot_absolute_error2d(times, func, benchmark, deriv_bench, equation, points_per_dim=25, outdir=None):
    prim_names = ('u', 'w', r'$\partial_x p$', r'$\partial_z p$', 'T')
    nrows = 5
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)

    # --Reshape arrays to match func in put dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))

    fig, ax = plt.subplots(nrows=nrows, ncols=len(times), figsize=(9, 10.5))
    title = f'PINN Error {equation}'
    fig.suptitle(title)

    for j in range(nrows):

        ax[j, 0].text(-0.6, 0.5, f'{prim_names[j]}', transform=ax[j, 0].transAxes, fontsize='x-large')

        for i, time in enumerate(times):

            if j == 0:
                ax[j, i].set_title(f't = {time}', y=0.99, fontsize='x-large')
            t = time * np.ones_like(x)
            xzt = np.hstack((x, z, t))
            match j:
                case 0 | 1:
                    out = np.abs(func(xzt) - benchmark(xzt))[:, j:j+1]
                case 2:
                    learned = func(xzt, operator=lambda x, y: dde.grad.jacobian(y, x, i=2, j=0))
                    true, _ = deriv_bench(xzt)
                    out = np.abs(learned - true)
                case 3:
                    learned = func(xzt, operator=lambda x, y: dde.grad.jacobian(y, x, i=2, j=1))
                    _, true = deriv_bench(xzt)
                    out = np.abs(learned - true)
                case 4:
                    out = np.abs(func(xzt) - benchmark(xzt))[:, j-1:j]
            Out = out.reshape(X.shape)
            cs = ax[j, i].contourf(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            ax[j, i].label_outer()

        fig.colorbar(cs, ax=ax[j, :])

    if outdir is not None:
        fig.savefig(f'{outdir}/absolute_error')
    else:
        plt.show()
