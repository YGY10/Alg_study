#include <chrono>
#include <iostream>

int main() {
    const int N = 100000000000;  // 循环次数
    volatile double x = 42.0;    // 防止编译器优化
    volatile double y;           // 存储结果

    // 测试乘法耗时
    auto start_mul = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        y = x + 2.0;
    }
    auto end_mul = std::chrono::high_resolution_clock::now();
    auto duration_mul =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_mul - start_mul).count();

    // 测试除法耗时
    auto start_div = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        y = x - 2.0;
    }
    auto end_div = std::chrono::high_resolution_clock::now();
    auto duration_div =
        std::chrono::duration_cast<std::chrono::milliseconds>(end_div - start_div).count();

    // 输出结果
    std::cout << "乘法耗时: " << duration_mul << " 毫秒" << std::endl;
    std::cout << "除法耗时: " << duration_div << " 毫秒" << std::endl;
    std::cout << "除法比乘法慢 " << (duration_div - duration_mul) << " 毫秒" << std::endl;

    return 0;
}