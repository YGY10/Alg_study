#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

// format: [capture] (parameters) -> return_type { function body }
int main() {
    int a = 1, b = 2;

    // 值捕获：复制一份 a 进去
    auto f1 = [a](int x) { return a + x + 10; };

    // 引用捕获：对 a 的引用，外面改了 a，里面能感知
    auto f2 = [&a]() { return a + 10; };

    // 混合捕获
    auto f3 = [a, &b]() { return a + b; };

    // 全部值捕获
    auto f4 = [=]() { return a + b; };

    // 全部引用捕获
    auto f5 = [&]() { return a + b; };

    std::vector<std::string> vec(3);
    vec[0] = "A";
    vec[1] = "B";
    vec[2] = "C";
    std::vector<int> vec2(3);
    std::transform(vec.begin(), vec.end(), vec2.begin(), f1);
    for (auto v : vec2) {
        std::cout << v << std::endl;
    }
}