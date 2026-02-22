#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <map>
#include <read_npz.h>
#include <fstream>


// CUDA kernel for Makima interpolation
__device__ double makima_interpolate_point(
    const double* x_data,
    const double* y_data,
    const double* slopes,
    int n,
    int chunk_size,
    const int* x_id,
    int* y_interp_shape,
    double x_interp
) {
    // Find the interval containing x_interp using binary search
    int left = 0, right = n - 1;
    int x_data_left  = x_id[3] * y_interp_shape[0],
        x_data_right = (x_id[3] + 1) * y_interp_shape[0] - 1,
        y_data_left  = x_id[2] * y_interp_shape[0] + x_id[3] * y_interp_shape[0] * y_interp_shape[2],
        y_data_right = (x_id[2] + 1) * y_interp_shape[0] + x_id[3] * y_interp_shape[0] * y_interp_shape[2] - 1;

    // Handle boundary cases
    if (x_interp <= x_data[x_data_left]) 
    {
//        left = 0, right = n - 1;
        x_data_left = x_data_left;
        x_data_right = x_data_left + 1;
        y_data_left = y_data_left;
        y_data_right = y_data_left + 1;
    } else if (x_interp >= x_data[x_data_right])
    {
//      left = n - 2;
//      right = n - 1;
        x_data_left = x_data_right - 1;
        x_data_right = x_data_right;
        y_data_left = y_data_right - 1;
        y_data_right = y_data_right;
    } else
    {
        // Binary search for the correct interval
//        while (right - left > 1) {
//            int mid = (left + right) / 2;
//            if (x_data[mid] <= x_interp) {
//                left = mid;
//            } else {
//                right = mid;
//            }
//        }
      while (x_data_right - x_data_left > 1)
      {
          int mid = (x_data_left + x_data_right) / 2;
          int mid_y  = (y_data_left + y_data_right) / 2;
          if (x_data[mid] <= x_interp)
          {
              x_data_left = mid;
              y_data_left = mid_y;
          } else
          {
              x_data_right = mid;
              y_data_right = mid_y;
          }
      }
    }

    // Get interval data
    double x0 = x_data[x_data_left];
    double x1 = x_data[x_data_right];
    double y0 = y_data[y_data_left];
    double y1 = y_data[y_data_right];
    double m0 = slopes[y_data_left];
    double m1 = slopes[y_data_right];

    // Normalize x to [0, 1] in the interval
    double h = x1 - x0;
    double t = (x_interp - x0) / h;

    // Hermite basis functions
    double h00 = 2*t*t*t - 3*t*t + 1;
    double h10 = t*t*t - 2*t*t + t;
    double h01 = -2*t*t*t + 3*t*t;
    double h11 = t*t*t - t*t;

    // Compute interpolated value
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
}

__global__ void makima_interpolate_kernel(
    const double* x_data,
    const double* y_data,
    const double* slopes,
    const double* x_interp,
    double* y_interp,
    int n_data,
    int n_interp,
    int* y_interp_shape,
    int chunk_size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_id[4], x_id_flat = 0;
    x_id[0] = (idx % y_interp_shape[0]);
    x_id[1] = (idx / y_interp_shape[0]) % y_interp_shape[1];
    x_id[2] = (idx / (y_interp_shape[0] * y_interp_shape[1])) % y_interp_shape[2];
    x_id[3] = (idx / (y_interp_shape[0] * y_interp_shape[1] * y_interp_shape[2])) % y_interp_shape[3];
    x_id_flat = x_id[0] + x_id[1] * y_interp_shape[0] + x_id[3] * y_interp_shape[0] * y_interp_shape[1];

    if (idx < n_interp)
    {
        y_interp[idx] = makima_interpolate_point(
            x_data, y_data, slopes, n_data, chunk_size, x_id, y_interp_shape, x_interp[x_id_flat]);
    }
}

// Host function to compute Makima slopes
void compute_makima_slopes(
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double>& slopes
) {
    int n = x.size();
    slopes.resize(n);

    if (n < 2) return;

    // Compute finite differences
    std::vector<double> dk(n + 3);

    // Interior differences
    for (int i = 0; i < n - 1; i++) {
        dk[i + 2] = (y[i + 1] - y[i]) / (x[i + 1] - x[i]);
    }

    // Boundary conditions using Makima's method
    if (n >= 3) {
        dk[1] = 2 * dk[2] - dk[3];
        dk[0] = 2 * dk[1] - dk[2];
        dk[n + 1] = 2 * dk[n] - dk[n - 1];
        dk[n + 2] = 2 * dk[n + 1] - dk[n];
    } else {
        dk[1] = dk[0] = dk[2];
        dk[n + 1] = dk[n + 2] = dk[2];
    }

    // Compute slopes using Makima's formula
    for (int i = 0; i < n; i++) {
        double w1 = std::abs(dk[i + 3] - dk[i + 2]);
        double w2 = std::abs(dk[i + 1] - dk[i]);

        if (w1 + w2 == 0) {
            slopes[i] = (dk[i + 1] + dk[i + 2]) * 0.5f;
        } else {
            slopes[i] = (w1 * dk[i + 1] + w2 * dk[i + 2]) / (w1 + w2);
        }
    }
}

void compute_makima_slopes(
    const std::vector<double>& x,
    const std::vector<double>& y,
    std::vector<double>& slopes,
    const uint32_t chunk_size,
    const uint32_t y_chunk_size
)
{
    // The chunk_size represents the least contiguous data which in XPSI terms is N phases.
    // y_chunk_size represents the photon energies
    int n = y.size(), n_chunks = n/chunk_size; // TODO: place a condition such that chunk size cannot be 0 here.
    slopes.resize(n);

    if (n < 2) return;

    // Compute finite differences
    std::vector<double> dk(n + 3 * n_chunks);

    uint32_t counter = 0, counter_y = 0, total_y = y_chunk_size * chunk_size, j = 0;
    // Interior differences
    for (int i = 0; i < n - 1; ++i)
    {
        counter = (int)(i / chunk_size);
        counter_y = (int)(i / total_y);
        j = ((int)(i % chunk_size)) + counter_y * chunk_size;
        dk[i + 2 + counter * 3] = (y[i + 1] - y[i]) / (x[j + 1] - x[j]);
    }

    std::printf("\nInterior differences done.\n");

    // Boundary conditions using Makima's method
    if (chunk_size >= 3)
    {
      for (uint64_t i = 0; i < n_chunks ; ++i)
      {
        dk[1 + i * (chunk_size + 3)] = 2 * dk[2 + i * (chunk_size + 3)] - dk[3 + i * (chunk_size + 3)];
        dk[0 + i * (chunk_size + 3)] = 2 * dk[1 + i * (chunk_size + 3)] - dk[2 + i * (chunk_size + 3)];
        dk[chunk_size + 1 + i * (chunk_size + 3)] = 2 * dk[chunk_size + i * (chunk_size + 3)] - dk[chunk_size - 1 + i *
          (chunk_size + 3)];
        dk[chunk_size + 2 + i * (chunk_size + 3)] = 2 * dk[chunk_size + 1 + i * (chunk_size + 3)] - dk[chunk_size + i *
          (chunk_size + 3)];
      }
    }
    else
    {
      for (uint64_t i = 0; i < n_chunks ; ++i)
      {
        dk[1 + i * (chunk_size + 3)] = dk[0 + i * (chunk_size + 3)] = dk[2 + i * (chunk_size + 3)];
        dk[chunk_size + 1 + i * (chunk_size + 3)] = dk[chunk_size + 2 + i * (chunk_size + 3)] = dk[2 + i * (chunk_size +
            3)];
      }
    }

    // Compute slopes using Makima's formula
    counter = 0;
    for (int i = 0; i < n; i++)
    {
        counter = (int)(i / chunk_size);
        double w1 = std::abs(dk[i + 3 + counter * 3] - dk[i + 2 + counter * 3]);
        double w2 = std::abs(dk[i + 1 + counter * 3] - dk[i + counter * 3]);

        if (w1 + w2 == 0)
        {
            slopes[i] = (dk[i + 1 + counter * 3] + dk[i + 2 + counter * 3]) * 0.5f;
        }
        else
        {
            slopes[i] = (w1 * dk[i + 1 + counter * 3] + w2 * dk[i + 2 + counter * 3]) / (w1 + w2);
        }
    }
}

class MakimaInterpolator {
private:
    double* d_x_data;
    double* d_y_data;
    double* d_slopes;
    int n_data;

public:
    MakimaInterpolator(const std::vector<double>& x, const std::vector<double>& y) {
        int nx_data = x.size();
        int ny_data = y.size();
        n_data  = nx_data;

        // #TODO: Horrible hard coding, derive this from the main function.
        // Represents size of photon energies and data array chunk_size
        uint32_t chunk_size = 30, y_chunk_size = 40;

        // Compute slopes on host
        std::vector<double> slopes;
        auto t_start = std::chrono::high_resolution_clock::now();
        compute_makima_slopes(x, y, slopes, chunk_size, y_chunk_size);

        // Allocate device memory
        cudaMalloc(&d_x_data, nx_data * sizeof(double));
        t_start = std::chrono::high_resolution_clock::now();
        cudaMalloc(&d_y_data, ny_data * sizeof(double));
        cudaMalloc(&d_slopes, ny_data * sizeof(double));

        // Copy data to device
        cudaMemcpy(d_x_data, x.data(), x.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_data, y.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slopes, slopes.data(), y.size() * sizeof(double), cudaMemcpyHostToDevice);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> s_double = t_end - t_start;
        double time_slope_calc = s_double.count();
        std::cout << "\n Time taken for slope calculation on the host and copy from host to device: " << time_slope_calc << " seconds.\n";

    }

    ~MakimaInterpolator() {
        cudaFree(d_x_data);
        cudaFree(d_y_data);
        cudaFree(d_slopes);
    }

    void interpolate(
        const std::vector<double>& x_interp,
        std::vector<double>& y_interp,
        const uint64_t n_interp
    ) {
        auto t_start = std::chrono::high_resolution_clock::now();
        y_interp.resize(n_interp);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> s_double = t_end - t_start;
        double time_mem_resize = s_double.count();
        std::cout << "\n Interpolation mem resize of the host mem took: " << time_mem_resize << " seconds.\n";

        // TODO: Horrible hard coding: get this from the main function
        // Array shape in reverese
        int y_interp_shape[4] = {30, 50, 40, 50};

        // Allocate device memory for interpolation points
        double* d_x_interp;
        double* d_y_interp;
        int* d_y_interp_shape;

        auto t1 = std::chrono::high_resolution_clock::now();
        cudaMalloc(&d_x_interp, x_interp.size() * sizeof(double));
        cudaMalloc(&d_y_interp, n_interp * sizeof(double));
        cudaMalloc(&d_y_interp_shape, 4 * sizeof(int));
        auto t2 = std::chrono::high_resolution_clock::now();

        s_double = t2 - t1;
        std::cout << "\n Interpolation mem alloc took: " << s_double.count() << " seconds.\n";

        // Copy interpolation points to device
        auto t3 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_x_interp, x_interp.data(), x_interp.size() * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_interp_shape, &y_interp_shape[0], 4 * sizeof(int), cudaMemcpyHostToDevice);
        auto t4 = std::chrono::high_resolution_clock::now();
        s_double = t4 - t3;
        std::cout << "\n Interpolation mem transfer H->D took: " << s_double.count() << " seconds.\n";

        // Launch kernel
        int block_size = 256;
        int grid_size = (n_interp + block_size - 1) / block_size;

        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();

        makima_interpolate_kernel<<<grid_size, block_size>>>(
            d_x_data, d_y_data, d_slopes, d_x_interp, d_y_interp, n_data, n_interp, d_y_interp_shape, 30
        );
        cudaDeviceSynchronize();
        t2 = std::chrono::high_resolution_clock::now();
        s_double = t2 - t1;

        std::cout << "\nThe actual interpolation without mem transfers took: " << s_double.count() << " seconds.\n";

        // Copy results back to host
        t1 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(y_interp.data(), d_y_interp, n_interp * sizeof(double), cudaMemcpyDeviceToHost);
        t2 = std::chrono::high_resolution_clock::now();
        s_double = t2 - t1;

        std::cout << "\n Interpolation mem transfer D->H took: " << s_double.count() << " seconds.\n";

        // Clean up
        t1 = std::chrono::high_resolution_clock::now();
        cudaFree(d_x_interp);
        cudaFree(d_y_interp);
        t2 = std::chrono::high_resolution_clock::now();
        s_double = t2 - t1;

        std::cout << "\n Interpolation mem freeing: " << s_double.count() << " seconds.\n";
        // Check for CUDA errors
        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        }
        t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> s_full_func = t_end - t_start;
        std::cout << "\n Interpolation full func: " << s_full_func.count() << " seconds.\n";
        std::cout << "\n Actual full interp time including mem alloc and transfers of the points to be interpolated: "
          << s_full_func.count() - time_mem_resize << " seconds.\n";
    }
};

// Example usage
int main() {
    py::scoped_interpreter guard{}; // Start the Python interpreter

    // Load the .npz file
    py::object np = py::module::import("numpy");
    py::object npz_file = np.attr("load")("/gpfs/home5/satishk/projects/promising_application/xpsi/interpolation/interpolation_products_reduced_correct_pulse.npz");

    printf("Loaded the npz file ... \n");
    std::vector<std::string> array_names;

    std::map<std::string, ArrayAttributes> array_attrs;

    // Sample data points
    // std::vector<double> x_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<double> x_data, y_data;
    // std::vector<double> y_data = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f}; // y = x^2
    // Points to interpolate
    std::vector<double> x_interp, y_check;
    // Get array names

    py::list files = npz_file.attr("files");
    double *data1;
    data1 = NULL;
    for (auto item : files)
    {
        std::string name = py::str(item);
        array_names.push_back(name);
        std::cout << name << std::endl;

        // Extract attributes for each array
        py::array arr = npz_file[py::str(name)].cast<py::array>();
        py::array_t<double> array1 = arr.cast<py::array_t<double>>();
        auto buf1 = array1.request();
        std::cout << " Size of this array : " << array1.size() << "\n";
        data1 = static_cast<double*>(buf1.ptr);
        array_attrs[name] = extract_attributes(name, arr);


        // Print and assign the trial data
        if (name == "phase_data_array")
        {
          for (size_t i = 0; i < 60; ++i)
          {
            std::cout << std::scientific;
            std::cout << data1[i] << " ";
            if ( !((i+1) % 5) )
              std::cout << "\n";
          }
          std::cout << std::endl;
          x_data.assign(data1, data1 + array1.size());
        }
        if (name == "intensity_data_array")
        {
          for (size_t i = 0; i < 60; ++i)
          {
            std::cout << std::scientific;
            std::cout << data1[i] << " ";
            if ( !((i+1) % 5) )
              std::cout << "\n";
          }
          std::cout << std::endl;
          y_data.assign(data1, data1 + array1.size());
        }
        if (name == "phase_query_array")
        {
          for (size_t i = 0; i < 30; ++i)
          {
            std::cout << std::scientific;
            std::cout << data1[i] << " ";
            if ( !((i+1) % 5) )
              std::cout << "\n";
          }
          std::cout << std::endl;
          x_interp.assign(data1, data1 + array1.size());
        }
        if (name == "intensity_query_array")
        {
          for (size_t i = 0; i < 30; ++i)
          {
            std::cout << std::scientific;
            std::cout << data1[i] << " ";
            if ( !((i+1) % 5) )
              std::cout << "\n";
          }
          std::cout << std::endl;
          y_check.assign(data1, data1 + array1.size());
        }
    }

    std::cout << "x Data, y Data, x Interp points, y Check points\n";
    for (size_t i = 0; i < 30; ++i)
    {
      std::cout << std::scientific;
      std::cout << x_data[i] << " ";
      if ( !((i+1) % 5) )
        std::cout << "\n";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 30; ++i)
    {
      std::cout << std::scientific;
      std::cout << y_data[i] << " ";
      if ( !((i+1) % 5) )
        std::cout << "\n";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 30; ++i)
    {
      std::cout << std::scientific;
      std::cout << x_interp[i] << " ";
      if ( !((i+1) % 5) )
        std::cout << "\n";
    }
    std::cout << std::endl;

    for (size_t i = 0; i < 30; ++i)
    {
      std::cout << std::scientific;
      std::cout << y_check[i] << " ";
      if ( !((i+1) % 5) )
        std::cout << "\n";
    }

    uint64_t num_elements = y_check.size(), interval = 10;
    // Create interpolator and calculate slopes
    MakimaInterpolator interp(x_data, y_data);

    std::printf("\nSlope calculation successful.\n");


    std::printf("\nGetting interpolation points ready ...\n");
//    double x  = 0.0f;
//    for (uint64_t i = 0; i <= num_elements; ++i)
//    {
//        if (i % interval == 0)
//          x += 1e-6;
//        x_interp.push_back(x);
//    }

    std::printf("\nPoints ready. Performing interpolation ...\n");
    // Perform interpolation
    std::vector<double> y_interp;
    // Measure timing
    auto t1 = std::chrono::high_resolution_clock::now();
    interp.interpolate(x_interp, y_interp, num_elements);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> s_double = t2 - t1;

    std::cout << std::endl;

    for (size_t i = 0; i < 30; ++i)
    {
      std::cout << std::scientific;
      std::cout << y_interp[i] << " ";
      if ( !((i+1) % 5) )
        std::cout << "\n";
    }

    // Print results 
    std::cout << "\nMakima Interpolation Results:\n";
    std::cout << "x\t\ty_interp\t\ty_actual\t\terror\t\trelative_error in percentage\n";
    std::cout << "----------------------------------------------------\n";

    for (size_t i = 1000; i < 1060; i++)
    {
      double x = x_interp[i];
      double y_actual = x * x; // True function
      double error = std::abs(y_interp[i] - y_check[i]);
      double rel_error = error / y_check[i] * 100;
      std::cout << x << "\t\t" << y_interp[i] << "\t\t" << y_check[i] << "\t\t" << error << "\t\t" <<
        rel_error << "\n";
    }

    // Output to a file
    double cum_percentage_err = 0, max_percentage_error = 0;;
    std::ofstream file_handle;
    file_handle.open("io_file.txt");
    for (size_t i = 0; i < y_interp.size(); ++i)
    {
      double error = std::abs(y_interp[i] - y_check[i]);
      double rel_error = error / y_check[i] * 100;
      cum_percentage_err += rel_error;
      max_percentage_error = (rel_error > max_percentage_error? rel_error : max_percentage_error);
      file_handle << std::scientific << y_interp[i] << "\t\t" << y_check[i] << "\t\t" << error << "\t\t" <<
        rel_error << "\t\t" << cum_percentage_err << "\t\t" << max_percentage_error << "\n";
    }
    file_handle.close();


    std::cout << "\nThe interpolation including HtoD and DtoH copy for " << num_elements << " points took: " << s_double.count() << " seconds.\n";

    return 0;
}
