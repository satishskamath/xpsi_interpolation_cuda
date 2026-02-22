# Sample command
# nvcc -I/sw/arch/RHEL9/EB_production/2025/software/Python/3.13.1-GCCcore-14.2.0/include/python3.13 -O3 -gencode arch=compute_90,code=sm_90a  -o interp_double_gencode_sm90a interp_double.cu -lpython3.13
# Required modules NVHPC (loads CUDA as well), SciPy-bundle and pybind11
# The include PATH for he python libraries and the linker library (specified by ld_libraries) needs to be changed accord
# to the Python module that is loaded.
#
# gencode and arch compiles for a specific GPU architecture (H100 in the case above), if compiling for any other GPU
# then that needs to be changed. This is done to prevent warmup time required for the JIT compilation when running for
# the first time.
#
# interp.cu -> FP32 version and interp_double.cu -> FP64 version
# NOTE: The FP32 version performs interpolation on a dummy function and not on the real data whereas the FP64 version
# uses the actual data provided by Bas via the npz files and requires interpolation_products_reduced_correct_pulse.npz.

files = interp.cu
files_double = interp_double.cu
cc = nvcc
exec = interp
exec_double = interp_double
python_include = /sw/arch/RHEL9/EB_production/2025/software/Python/3.13.1-GCCcore-14.2.0/include/python3.13
cc_flags = -O3 -gencode arch=compute_90,code=sm_90a
ld_libraries = -lpython3.13

all: $(files)
	$(cc) -I$(python_include) $(cc_flags) -o $(exec) $(files) $(ld_libraries)
	$(cc) -I$(python_include) $(cc_flags) -o $(exec_double) $(files_double) $(ld_libraries)

clean:
	rm -rf $(exec)
	rm -rf $(exec_double)
