#include <cuda_runtime.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

void test_cuda_with_env() {
    std::cout << "=== 带环境设置的CUDA测试 ===" << std::endl;

    // 1. 检查CUDA运行时
    int cuda_device_count = 0;
    cudaError_t cuda_status = cudaGetDeviceCount(&cuda_device_count);
    std::cout << "CUDA Runtime设备数: " << cuda_device_count << std::endl;
    std::cout << "CUDA Runtime状态: "
              << (cuda_status == cudaSuccess ? "成功" : cudaGetErrorString(cuda_status))
              << std::endl;

    // 2. 检查PyTorch CUDA
    std::cout << "PyTorch CUDA可用: " << torch::cuda::is_available() << std::endl;
    std::cout << "PyTorch CUDA设备数: " << torch::cuda::device_count() << std::endl;

    // 3. 尝试直接使用CUDA设备
    if (cuda_device_count > 0) {
        try {
            std::cout << "尝试直接创建CUDA设备..." << std::endl;

            // 方法1: 使用设备构造函数
            torch::Device cuda_device(torch::kCUDA, 0);
            std::cout << "CUDA设备对象创建成功" << std::endl;

            // 方法2: 在CPU上创建然后移动到GPU
            torch::Tensor cpu_tensor = torch::ones({2, 2});
            std::cout << "CPU张量: " << cpu_tensor.device() << std::endl;

            torch::Tensor gpu_tensor = cpu_tensor.to(cuda_device);
            std::cout << "GPU张量: " << gpu_tensor.device() << std::endl;

            // 方法3: 直接在GPU上创建
            auto options = torch::TensorOptions().device(torch::kCUDA);
            torch::Tensor direct_gpu_tensor = torch::ones({2, 2}, options);
            std::cout << "直接GPU张量: " << direct_gpu_tensor.device() << std::endl;

            std::cout << "✅ CUDA测试全部成功!" << std::endl;

        } catch (const std::exception& e) {
            std::cout << "❌ CUDA测试失败: " << e.what() << std::endl;
        }
    }
}

int main() {
    test_cuda_with_env();
    return 0;
}