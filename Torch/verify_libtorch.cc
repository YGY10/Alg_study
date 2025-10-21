#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>
#include <torch/torch.h>
#include <iostream>

int main() {
    try {
        // 手动触发 CUDA 初始化
        if (!torch::cuda::is_available()) {
            std::cout << "⚠️ CUDA 未初始化，尝试强制初始化..." << std::endl;
            c10::cuda::CUDACachingAllocator::emptyCache();  // 触发初始化
        }

        std::cout << "torch::cuda::is_available() = " << torch::cuda::is_available() << std::endl;
        std::cout << "CUDA 设备数: " << torch::cuda::device_count() << std::endl;

        if (torch::cuda::is_available()) {
            auto props = at::cuda::getCurrentDeviceProperties();
            std::cout << "当前 CUDA 设备: " << props->name << std::endl;
        }

        // 测试 GPU 张量创建
        torch::Tensor t = torch::ones({2, 2}, torch::TensorOptions().device(torch::kCUDA));
        std::cout << "GPU 张量创建成功: " << t.device() << std::endl;

        std::cout << "✅ CUDA 环境正常" << std::endl;

        // ------------------------------
        // ✅ 加载 TorchScript 模型并推理
        // ------------------------------
        torch::jit::script::Module module;  // ✅ 声明模型对象
        module = torch::jit::load("transformer_regressor.pt");
        module.to(torch::kCUDA);
        std::cout << "✅ 模型加载成功" << std::endl;

        torch::Tensor input = torch::randn({1, 10, 1}, torch::TensorOptions().device(torch::kCUDA));
        auto output = module.forward({input}).toTensor();
        std::cout << "输出: " << output << std::endl;

    } catch (const std::exception &e) {
        std::cerr << "❌ 运行错误: " << e.what() << std::endl;
    }

    return 0;
}
