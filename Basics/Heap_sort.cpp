// 堆排序（Heap sort）
// 核心思想：利用堆这种数据结构的特性，将数组转换为最大堆或最小堆，然后逐步取出最大或最小元素
// 适合大规模数据排序，时间复杂度为O(n log n)，空间复杂度为O(1)
// 注意：堆排序不是稳定的排序算法
#include <iostream>
#include <vector>

// 下城调整，构建最大堆
void heapify(std::vector<double>& orign_vector, int n, int i) {
    int largest = i;  // 初始化最大元素为根节点
    int left = 2 * i + 1;
    int right = 2 * i + 2;
    if (left < n && orign_vector[left] > orign_vector[largest]) {
        largest = left;  // 如果左子节点大于根节点，更新最大元素
    }
    if (right < n && orign_vector[right] > orign_vector[largest]) {
        largest = right;  // 如果右子节点大于当前最大元素，更新最大元素
    }

    if (largest != i) {
        std::swap(orign_vector[i], orign_vector[largest]);  // 交换根节点和最大元素
        heapify(orign_vector, n, largest);                  // 递归调整子树
    }
}

// 最大堆
void Heap_sort(std::vector<double>& orign_vector) {
    int n = orign_vector.size();
    for (int i = n / 2 - 1; i >= 0; i--) {
        heapify(orign_vector, n, i);  // 构建最大堆
    }
    // 逐步取出最大值放到末尾
    for (int i = n - 1; i > 0; i--) {
        std::swap(orign_vector[0], orign_vector[i]);  // 将最大元素放到数组末尾
        heapify(orign_vector, i, 0);                  // 调整剩余元素，重新构建最大堆
    }
}

int main() {
    std::vector<double> nums = {3.2, 1.5, 4.8, 2.0, 0.9};
    std::cout << "Before sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    Heap_sort(nums);
    std::cout << "After sorting: ";
    for (double num : nums) std::cout << num << " ";
    std::cout << std::endl;
    return 0;
}