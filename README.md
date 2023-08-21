# PINN-PE

Numerical component of
Hu, R., Lin, Q., Raydan, A. et al. Higher-order error estimates for physics-informed neural networks approximating the primitive equations. Partial Differ. Equ. Appl. 4, 34 (2023). https://doi.org/10.1007/s42985-023-00254-y

---
You will need to install PyTorch and DeepXDE. If you are using a conda environment you can install both with the following commands:
```shell script
conda install pytorch
conda install -c conda-forge deepxde
```

After both are installed, you will need to set DeepXDE's backend to `pytorch` in order for it to know that it should use PyTorch for it's neural networks.
Consult the DeepXDE documentation for how to do this.

---
## Program Entry Point
You can run the program in `main.py` by passing the equation number of the benchmark equation from section 5
that you wish to train as a command line argument. For example,
```shell script
python3 main.py 5.2
```
will train the PINN on the initial and boundary data given from equation 5.2.

The program will attempt to execute two runs in parallel; one trained using the $L^2$ residuals
and the other using the $H^1$ residuals.
