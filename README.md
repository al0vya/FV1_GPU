# FV1-GPU

## Model description

This is a successor of <a href="https://github.com/al0vya/FV1_cpp">FV1-CPU</a> that employs the finite volume (FV1) scheme to solve the one-dimensional shallow water equations in parallel on single NVIDIA GPUs using CUDA, called 'FV1-GPU'.

## Running the model

FV1-GPU requires having the CUDA Toolkit installed. It has been developed using CUDA Toolkit 10.0+ and on a GPU of <a href="https://en.wikipedia.org/wiki/CUDA#Version_features_and_specifications">Compute Capability 6.1</a>; no other Compute Capabilities have been tested. It has the same 6 test cases as FV1-CPU. After building and then running the executable, the user must select:

* Which test case to run
* How many cells are to comprise the mesh

## Speedups relative to FV1-CPU

Below are speedups of FV1-GPU over FV1-CPU for a 40 s wet dam break simulation, run in release mode on a machine with an i5-8250U CPU and GTX 1050M GPU. L dictates the number of cells in the mesh as 2<sup>L</sup>.

![speedup](/FV1_GPU_1D/parallelisation_speedup.png)
