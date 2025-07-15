// 冒泡排序
// 为什么冒泡排序是稳定的？
// 因为冒泡排序在比较两个相邻元素时，如果他们相等，不会交换位置，所以保持了他们原始相对顺序
#include <iostream>
#include <vector>

void Bubble_sort(std::vector<double>& orign_vector) {
    int n = orign_vector.size();
    bool swapped = false;
    for (int i = 0; i < n - 1; i++) {
        swapped = false;
        for (int j = 0; j < n - 1 - i; j++) {
            if (orign_vector[j] < orign_vector[j + 1]) {
                std::swap(orign_vector[j], orign_vector[j + 1]);
                swapped = true;
            }
        }
        if (!swapped) break;
    }
}

int main() {
    std::vector<double> nums = {3.2, 1.5, 4.8, 2.0, 0.9};
    std::cout << "Before sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    Bubble_sort(nums);
    std::cout << "After sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    return 0;
}