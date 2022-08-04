import argparse
import numpy as np
import matplotlib.pyplot as plt


def get_params():
    """
    Function for parsing parameters from config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='Directory to output data.')
    args = parser.parse_args()
    return args.outdir


def plot_all_output3d(times, func, points_per_dim=25, outdir=None):
    prim_names = ('u', 'w', 'p', 'T')
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)

    # --Reshape arrays to match func input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))

    fig, ax = plt.subplots(nrows=4, ncols=len(times), figsize=(9, 9.5), subplot_kw=dict(projection='3d'))
    title = func.__name__
    if func.__name__ == 'predict':
        title = 'PINN Output'
    fig.suptitle(title)

    for j in range(4):

        maximum = -np.inf
        minimum = np.inf

        for i, time in enumerate(times):

            t = time * np.ones_like(x)
            xzt = np.hstack((x, z, t))
            out = func(xzt)
            maximum = np.max(out[:, j]) if np.max(out[:, j]) > maximum else maximum
            minimum = np.min(out[:, j]) if np.min(out[:, j]) < minimum else minimum
            Out = out[:, j].reshape(X.shape)
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

    fig.subplots_adjust(0.05, 0.1, 0.92, 0.92)

    if outdir is not None:
        fig.savefig(f'{outdir}/model_output')
    else:
        plt.show()


def plot_error2d(times, func, benchmark, error, points_per_dim=25, outdir=None):
    assert error == 'absolute' or error == 'relative'
    prim_names = ('u', 'w', 'p', 'T')
    x_vals = np.linspace(0.0, 1.0, points_per_dim)
    z_vals = np.linspace(0.0, 1.0, points_per_dim)

    # --Reshape arrays to match func input dims--
    X, Z = np.meshgrid(x_vals, z_vals)
    x = X.reshape((-1, 1))
    z = Z.reshape((-1, 1))

    fig, ax = plt.subplots(nrows=4, ncols=len(times), figsize=(8, 7.5))
    title = f'{func.__name__} error'
    if func.__name__ == 'predict':
        title = 'PINN Error'
    fig.suptitle(title)

    # TODO: change `range(4)` for better generalization
    for j in range(4):

        ax[j, 0].text(-0.6, 0.5, f'{prim_names[j]}', transform=ax[j, 0].transAxes, fontsize='x-large')

        for i, time in enumerate(times):

            if j == 0:
                ax[j, i].set_title(f't = {time}', y=0.99, fontsize='x-large')
            t = time * np.ones_like(x)
            xzt = np.hstack((x, z, t))
            if error == 'absolute':
                out = np.abs(func(xzt) - benchmark(xzt))
            if error == 'relative':
                out = np.abs((benchmark(xzt) - func(xzt)) / benchmark(xzt))
            Out = out[:, j].reshape(X.shape)
            cs = ax[j, i].contourf(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            ax[j, i].label_outer()

        fig.colorbar(cs, ax=ax[j, :])

    if outdir is not None:
        fig.savefig(f'{outdir}/{error}_error')
    else:
        plt.show()
