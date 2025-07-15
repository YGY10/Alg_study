// 要求：给定一个有序数组和一个目标值，返回目标值在数组中的索引，如果不存在则返回 -1。
#include <iostream>
#include <vector>

int binary_search(const std::vector<int>& nums, int target) {
    int left = 0;
    int right = nums.size() - 1;
    while (left <= right) {
        int mid_index = left + (right - left) / 2;
        int mid_value = nums[mid_index];
        if (mid_value < target) {
            left = mid_index + 1;
        }
        if (mid_value > target) {
            right = mid_index - 1;
        }
        if (mid_value == target) {
            return mid_index;
        }
    }
    return -1;
}

int main() {
    std::vector<int> nums = {1, 2, 3, 4, 5, 6};
    int target = 4;
    int position = binary_search(nums, target);
    std::cout << "find position " << position << std::endl;
}