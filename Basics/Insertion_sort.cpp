// 插入排序(Insertion sort)
// 核心思想：把当前元素插入到它前面已经排好序的序列中，类似玩扑克牌按顺序插进手牌一样
// 适合小规模数据排序或近似有序的数组
#include <iostream>
#include <vector>

void Insertion_sort(std::vector<double>& orign_vector) {
    int n = orign_vector.size();
    for (int i = 1; i < n; i++) {
        double key = orign_vector[i];
        int j = i - 1;
        while (j >= 0 && orign_vector[j] > key) {
            orign_vector[j + 1] = orign_vector[j];
            j--;
        }
        orign_vector[j + 1] = key;
    }
}

int main() {
    std::vector<double> nums = {3.2, 1.5, 4.8, 2.0, 0.9};
    std::cout << "Before sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    Insertion_sort(nums);
    std::cout << "After sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
}