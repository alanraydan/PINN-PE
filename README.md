# PINN-PE

Numerical component of "Error Estimates for Physics Informed Neural 
Networks Approximating the Primitive Equations" by Ruimeng Hu, Quyuan Lin, 
Alan Raydan, and Sui Tang.

---
You will need to install PyTorch and DeepXDE. If you are using a conda environment you can install both with the following commands:
```shell script
conda install pytorch
conda install -c conda-forge deepxde
```

After both are installed, you will need to set DeepXDE's backend to `pytorch` in order for it to know that it should use PyTorch for it's neural networks.
Consult the DeepXDE documentation for how to do this.

---
If you run into the following error when you try to run the newest version of the code, let me know and I can help fix it:
```
RuntimeError: expected scalar type Float but found Double
```
