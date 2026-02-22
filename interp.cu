#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>

// CUDA kernel for Makima interpolation
__device__ float makima_interpolate_point(
    const float* x_data, 
    const float* y_data, 
    const float* slopes,
    int n,
    int idx,
    const int* x_id,
    float x_interp
) {
    // Find the interval containing x_interp using binary search
    int left = 0, right = n - 1;
    if (idx == 10 || idx == 1001)
    {
      std::printf("Device code \n%d %d\n", idx, n);
      std::printf("Device code \n%d %d %d %d\n", x_id[0], x_id[1], x_id[2], x_id[3]);
    }

    // Handle boundary cases
    if (x_interp <= x_data[0]) {
        left = 0;
        right = 1;
    } else if (x_interp >= x_data[n-1]) {
        left = n - 2;
        right = n - 1;
    } else {
        // Binary search for the correct interval
        while (right - left > 1) {
            int mid = (left + right) / 2;
            if (x_data[mid] <= x_interp) {
                left = mid;
            } else {
                right = mid;
            }
        }
    }
    
    // Get interval data
    float x0 = x_data[left];
    float x1 = x_data[right];
    float y0 = y_data[left];
    float y1 = y_data[right];
    float m0 = slopes[left];
    float m1 = slopes[right];
    
    // Normalize x to [0, 1] in the interval
    float h = x1 - x0;
    float t = (x_interp - x0) / h;
    
    // Hermite basis functions
    float h00 = 2*t*t*t - 3*t*t + 1;
    float h10 = t*t*t - 2*t*t + t;
    float h01 = -2*t*t*t + 3*t*t;
    float h11 = t*t*t - t*t;
    
    // Compute interpolated value
    return h00 * y0 + h10 * h * m0 + h01 * y1 + h11 * h * m1;
}

__global__ void makima_interpolate_kernel(
    const float* x_data,
    const float* y_data,
    const float* slopes,
    const float* x_interp,
    float* y_interp,
    int n_data,
    int n_interp
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int x_id[4] = {idx, idx+1, idx+2, n_data};
    if (idx == 10 || idx == 1001)
      std::printf("Global func code \n%d %d\n", idx, n_data);
    if (idx < n_interp) {
        y_interp[idx] = makima_interpolate_point(
            x_data, y_data, slopes, n_data, idx, x_id, x_interp[idx]
        );
    }
}

// Host function to compute Makima slopes
void compute_makima_slopes(
    const std::vector<float>& x,
    const std::vector<float>& y,
    std::vector<float>& slopes
) {
    int n = x.size();
    slopes.resize(n);
    
    if (n < 2) return;
    
    // Compute finite differences
    std::vector<float> dk(n + 3);
    
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
        float w1 = std::abs(dk[i + 3] - dk[i + 2]);
        float w2 = std::abs(dk[i + 1] - dk[i]);
        
        if (w1 + w2 == 0) {
            slopes[i] = (dk[i + 1] + dk[i + 2]) * 0.5f;
        } else {
            slopes[i] = (w1 * dk[i + 1] + w2 * dk[i + 2]) / (w1 + w2);
        }
    }
}

class MakimaInterpolator {
private:
    float* d_x_data;
    float* d_y_data;
    float* d_slopes;
    int n_data;
    
public:
    MakimaInterpolator(const std::vector<float>& x, const std::vector<float>& y) {
        n_data = x.size();
        
        // Compute slopes on host
        std::vector<float> slopes;
        auto t_start = std::chrono::high_resolution_clock::now();
        compute_makima_slopes(x, y, slopes);
        
        // Allocate device memory
        cudaMalloc(&d_x_data, n_data * sizeof(float));
        cudaMalloc(&d_y_data, n_data * sizeof(float));
        cudaMalloc(&d_slopes, n_data * sizeof(float));
        t_start = std::chrono::high_resolution_clock::now();
        // Copy data to device
        cudaMemcpy(d_x_data, x.data(), n_data * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_y_data, y.data(), n_data * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_slopes, slopes.data(), n_data * sizeof(float), cudaMemcpyHostToDevice);
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
        const std::vector<float>& x_interp,
        std::vector<float>& y_interp
    ) {
        auto t_start = std::chrono::high_resolution_clock::now();
        int n_interp = x_interp.size();
        y_interp.resize(n_interp);
        auto t_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> s_double = t_end - t_start;
        double time_mem_resize = s_double.count();
        std::cout << "\n Interpolation mem resize of the host mem took: " << time_mem_resize << " seconds.\n";
        
        // Allocate device memory for interpolation points
        float* d_x_interp;
        float* d_y_interp;
        
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaMalloc(&d_x_interp, n_interp * sizeof(float));
        cudaMalloc(&d_y_interp, n_interp * sizeof(float));
        auto t2 = std::chrono::high_resolution_clock::now();
        
        s_double = t2 - t1;
        std::cout << "\n Interpolation mem alloc took: " << s_double.count() << " seconds.\n";
        
        // Copy interpolation points to device
        auto t3 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(d_x_interp, x_interp.data(), n_interp * sizeof(float), cudaMemcpyHostToDevice);
        auto t4 = std::chrono::high_resolution_clock::now();
        s_double = t4 - t3;
        std::cout << "\n Interpolation mem transfer H->D took: " << s_double.count() << " seconds.\n";
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (n_interp + block_size - 1) / block_size;
        
        t1 = std::chrono::high_resolution_clock::now();
        
        makima_interpolate_kernel<<<grid_size, block_size>>>(
            d_x_data, d_y_data, d_slopes, d_x_interp, d_y_interp, n_data, n_interp
        );
        t2 = std::chrono::high_resolution_clock::now();
        s_double = t2 - t1;
        
        std::cout << "\nThe actual interpolation without mem transfers took: " << s_double.count() << " seconds.\n";
        
        // Copy results back to host
        t1 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(y_interp.data(), d_y_interp, n_interp * sizeof(float), cudaMemcpyDeviceToHost);
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
    // Sample data points
    std::vector<float> x_data = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
    std::vector<float> y_data = {0.0f, 1.0f, 4.0f, 9.0f, 16.0f, 25.0f}; // y = x^2
    
    uint64_t num_elements = 3.2e6, interval = 10;
    // Create interpolator
    MakimaInterpolator interp(x_data, y_data);
    

    std::printf("\nGetting interpolation points ready ...\n");
    // Points to interpolate
    std::vector<float> x_interp;
    float x  = 0.0f;
    for (uint64_t i = 0; i <= num_elements; ++i)
    {
        if (i % interval == 0)
          x += 1e-6;
        x_interp.push_back(x);
    }

    std::printf("\nPoints ready. Performing interpolation ...\n");
    // Perform interpolation
    std::vector<float> y_interp;
    // Measure timing
    auto t1 = std::chrono::high_resolution_clock::now();
    interp.interpolate(x_interp, y_interp);
    auto t2 = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> s_double = t2 - t1;
    
    // Print results
    std::cout << "\nMakima Interpolation Results:\n";
    std::cout << "x\t\ty_interp\t\ty_actual\t\terror\n";
    std::cout << "----------------------------------------------------\n";
    
//    for (size_t i = 0; i < x_interp.size(); i++) {
//        float x = x_interp[i];
//        float y_actual = x * x; // True function
//        float error = std::abs(y_interp[i] - y_actual);
//        
//        std::printf("%.1f\t\t%.6f\t\t%.6f\t\t%.6f\n", 
//                   x, y_interp[i], y_actual, error);
//    }

    std::cout << "\nThe interpolation including HtoD and DtoH copy for " << num_elements << " points took: " << s_double.count() << " seconds.\n";
    
    return 0;
}
