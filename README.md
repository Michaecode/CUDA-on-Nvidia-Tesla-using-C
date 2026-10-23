# CUDA N-body Gravity Simulator

A CUDA C++ gravitational N-body simulation. Each CUDA thread computes the acceleration acting on one body by summing the gravitational contribution of every other body. A second kernel integrates velocities and positions.

## Features

- Thousands of mutually attracting bodies.
- One CUDA thread per body for force calculation.
- Newtonian gravity with softening to avoid singular forces at very small distances.
- Semi-implicit (symplectic) Euler integration.
- Deterministic initial conditions, configurable body count, step count, and timestep.
- GPU timing using CUDA events and optional final-state CSV export.

## Hardware and software

The original lab environment targets an NVIDIA Tesla K20Xm (Compute Capability 3.5). Use a CUDA Toolkit version that still supports `sm_35` (for example, CUDA 10.2). Newer toolkits may have removed Kepler code-generation support even if the installed NVIDIA driver is newer.

## Build

```bash
make
```

The default target is `sm_35` for the Tesla K20Xm. For another GPU, override it, for example:

```bash
make CUDA_ARCH=sm_75
```

Direct compilation:

```bash
nvcc -O3 -lineinfo -arch=sm_35 nbody.cu -o nbody
```

## Run

```bash
./nbody
```

Defaults to 4,096 bodies and 100 simulation steps.

```text
./nbody [bodies] [steps] [dt] [output.csv]
```

Examples:

```bash
./nbody 2048 100 0.001
./nbody 8192 200 0.0005 final-state.csv
```

When a CSV path is supplied, the final state is written with columns `x,y,z,vx,vy,vz,mass`. The reported time measures the GPU kernel loop and excludes initialization, memory allocation, and the final device-to-host copy.

## Complexity and numerical notes

The direct all-pairs method performs `O(N^2)` pair interactions per step. It is useful for learning GPU parallelism but gets expensive at large particle counts. Softening and timestep are fixed constants near the top of `nbody.cu`; reduce the timestep if the simulation becomes unstable. This is an educational simulator, not a high-precision astrophysics integrator.

## Troubleshooting

- **Unsupported GPU architecture:** install a CUDA Toolkit that supports your GPU's compute capability, or set `CUDA_ARCH` to an architecture supported by your target GPU and toolkit.
- **No CUDA compiler:** install the CUDA Toolkit and ensure `nvcc` is on your `PATH`.
- **Slow at high body counts:** the direct method calculates all pairwise forces and scales quadratically.
