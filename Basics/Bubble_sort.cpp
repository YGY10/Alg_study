#include <iostream>
#include <vector>

void Bubble_sort(std::vector<double>& orign_vector) {
    int n = orign_vector.size();
    for (int i = 0; i < n - 1; i++) {
        bool is_swapp = false;
        for (int j = 0; j < n - i - 1; j++) {
            if (orign_vector[j] > orign_vector[j + 1]) {
                std::swap(orign_vector[j], orign_vector[j + 1]);
                is_swapp = true;
            }
        }
        if (!is_swapp) break;
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