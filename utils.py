import argparse


def get_params():
    """
    Function for parsing parameters from config file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('outdir', help='Directory to output data.')
    args = parser.parse_args()
    return args.outdir
