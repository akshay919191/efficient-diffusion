#include <torch/extension.h>

void launch_welford_kernel(const torch::Tensor& A, torch::Tensor& mean_out, torch::Tensor& var_out);

void welford_forward(torch::Tensor A, torch::Tensor mean_out, torch::Tensor var_out) {
    
    TORCH_CHECK(A.is_cuda(), "Tensor A must be a CUDA tensor");
    TORCH_CHECK(A.is_contiguous(), "Tensor A must be contiguous");
    TORCH_CHECK(A.scalar_type() == torch::kHalf, "Tensor A must be FP16 (half)");

    launch_welford_kernel(A, mean_out, var_out);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &welford_forward, "Welford Online Variance and Mean (CUDA)");
}