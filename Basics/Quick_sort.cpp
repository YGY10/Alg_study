// 快速排序（Quicksort）采用分治法：
// 1.从数组中选一个基准值
// 2.把所有小于它的元素放到左边，大于它的放右边
// 3.对左右两边的子数组递归地快速排序
// 时间复杂度 O(n log n) 空间复杂度 O(log n)
#include <iostream>
#include <vector>

void Quick_sort(std::vector<double> &orign_vector, int left, int right) {
    if (orign_vector.empty() || left >= right) return;
    int i = left, j = right;
    double basis = orign_vector[left];
    while (i < j) {
        if (i < j && orign_vector[j] >= basis) {
            j--;
        }
        if (i < j && orign_vector[i] <= basis) {
            i++;
        }
        if (i < j) std::swap(orign_vector[i], orign_vector[j]);
    }
    std::swap(orign_vector[left], orign_vector[i]);
    Quick_sort(orign_vector, left, j - 1);
    Quick_sort(orign_vector, j + 1, right);
}

int main() {
    std::vector<double> nums = {3.2, 1.5, 4.8, 2.0, 0.9};
    std::cout << "Before sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    Quick_sort(nums, 0, nums.size() - 1);
    std::cout << "After sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    return 0;
}
