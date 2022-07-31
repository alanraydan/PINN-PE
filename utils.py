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
    if outdir is not None:
        fig.savefig(f'{outdir}/model_output')
    else:
        plt.show()


def plot_error2d(times, func, benchmark, error, points_per_dim=25, outdir=None):
    assert error == 'absolute' or error == 'relative'
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
        if error == 'absolute':
            out = np.abs(func(xzt) - benchmark(xzt))
        if error == 'relative':
            out = np.abs((benchmark(xzt) - func(xzt)) / benchmark(xzt))
        for j in range(out.shape[1]):
            Out = out[:, j].reshape(X.shape)
            cs = ax[j, i].contourf(X, Z, Out)
            ax[j, i].set_xlabel('x')
            ax[j, i].set_ylabel('z')
            fig.colorbar(cs, ax=ax[j, i])
    if outdir is not None:
        fig.savefig(f'{outdir}/{error}_error')
    else:
        plt.show()
