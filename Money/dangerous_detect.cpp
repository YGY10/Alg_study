#include <Python.h>
#include <iostream>
#include <vector>
#include "include/matplotlibcpp.h"

namespace plt = matplotlibcpp;

int main() {
    std::cout << "Matplotlib-cpp test start\n";

    // 禁止加载 ~/.local 中的脏包
    setenv("PYTHONNOUSERSITE", "1", 1);

    // 设置 conda 的 Python 环境路径
    setenv("PYTHONHOME", "/home/yihang/miniconda3/envs/cpppython", 1);
    setenv("PYTHONPATH", "/home/yihang/miniconda3/envs/cpppython/lib/python3.8/site-packages", 1);

    std::cout << "Calling interpreter...\n";

    // 让 matplotlibcpp 自动初始化 Python
    auto& interp = plt::detail::_interpreter::get();

    // 打印 Python 的 sys.path（验证是否干净）
    PyRun_SimpleString(R"(
import sys
print("Final sys.path =", sys.path)
)");

    std::vector<double> x = {0, 1, 2, 3, 4};
    std::vector<double> y = {0, 1, 0, 1, 0};
    plt::plot(x, y);
    plt::show();
}
