#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

void softmax1(const std::vector<float> &input, std::vector<float> &output){
    output.resize(input.size());  // ← 必须是 resize
    float max_input_val = *std::max_element(input.begin(), input.end());
    float sum = 0.f;

    for (float val : input) {
        sum += std::exp(val - max_input_val);
    }
    for (int i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i] - max_input_val) / sum;
    }
}

void softmax2(const std::vector<float> &input, std::vector<float> &output) {
    output.resize(input.size());  // ← 必须是 resize
    float sum = 0.f;

    for (float val : input) {
        sum += std::exp(val);
    }
    for (int i = 0; i < input.size(); i++) {
        output[i] = std::exp(input[i]) / sum;
    }
}

int main() {
    std::vector<float> x1 = {1.0f, 2.0f, 3.0f, 4.0f};
    std::vector<float> x2 = {1000.f, 1001.f, 1002.f};

    for (const auto &x : {x1, x2}) {
        std::vector<float> y1, y2;
        softmax1(x, y1);
        softmax2(x, y2);

        std::cout << "Input: ";
        for (float v : x) std::cout << v << ' ';
        std::cout << "\nsoftmax1 (stable): ";
        for (float v : y1) std::cout << std::fixed << std::setprecision(6) << v << ' ';
        std::cout << "\nsoftmax2 (naive):  ";
        for (float v : y2) std::cout << v << ' ';
        std::cout << "\n\n";
    }
    return 0;
}
