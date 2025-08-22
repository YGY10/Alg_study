#include <algorithm>
#include <vector>
#include <iostream>
#include <algorithm>

int main() {
    std::vector<int> vec = {1, 2, 3, -4, 5};
    int find_target = -4;
    bool result =
    std::any_of(vec.begin(), vec.end(), [find_target](int x) { return x == find_target; });
    std::cout << result << std::endl;

}