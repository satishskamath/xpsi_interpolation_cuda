# xpsi_interpolation_cuda
This code is the CUDA implementation of the 1-D Makima interpolation as required by the XPSI code

## Steps before compilation
- Clone the repository.
- Change the PATH to the npz file in `interp_double.cu`, pointing to `interpolation_products_reduced_correct_pulse.npz`.

```C++
    // Load the .npz file
    py::object np = py::module::import("numpy");
    py::object npz_file = np.attr("load")("./interpolation_products_reduced_correct_pulse.npz");

```
- If relative PATH does not work, then put in the full PATH to the file.

## Compilation
- Check the `Makefile` and make the changes below if and as needed.
- Sample command:
```bash
nvcc -I/sw/arch/RHEL9/EB_production/2025/software/Python/3.13.1-GCCcore-14.2.0/include/python3.13 -O3 -gencode arch=compute_90,code=sm_90a  -o interp_double_gencode_sm90a interp_double.cu -lpython3.13
```
- Required modules: NVHPC (loads CUDA as well), SciPy-bundle and pybind11.
- The include PATH for he python libraries and the linker library (specified by `ld_libraries`) needs to be changed
  according to the Python module that is loaded.
- gencode and arch compiles for a specific GPU architecture (H100 in the case above), if compiling for any other GPU
  then that needs to be changed. This is done to prevent warmup time required for the JIT compilation when running for
  the first time.
- `interp.cu` -> FP32 version and `interp_double.cu` -> FP64 version
- NOTE: The FP32 version performs interpolation on a dummy function and not on the real data whereas the FP64 version
  uses the actual data provided by Bas via the npz files and requires
  `interpolation_products_reduced_correct_pulse.npz`.

## Execution

- FP64:
```bash
./interp_double
```

- FP32:
```bash
./interp
```

## Raw timings

- H100 GPU:
```
Interior differences done.

 Time taken for slope calculation on the host and copy from host to device: 2.277790e-04 seconds.

Slope calculation successful.

Getting interpolation points ready ...

Points ready. Performing interpolation ...

 Interpolation mem resize of the host mem took: 2.405786e-03 seconds.

 Interpolation mem alloc took: 1.276100e-04 seconds.

 Interpolation mem transfer H->D took: 5.233900e-05 seconds.

The actual interpolation without mem transfers took: 1.654090e-04 seconds.

 Interpolation mem transfer D->H took: 9.454250e-04 seconds.

 Interpolation mem freeing: 1.305890e-04 seconds.

 Interpolation full func: 3.888038e-03 seconds.

 Actual full interp time including mem alloc and transfers of the points to be interpolated: 1.482252e-03 seconds.

...

The interpolation including HtoD and DtoH copy for 3000000 points took: 3.896378e-03 seconds.
```
