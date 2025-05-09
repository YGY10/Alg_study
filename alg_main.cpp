#include <iostream>
#include "A_Star.h"

int main() {
    GridMap grid_map(3, 3);
    Node* start = new Node(2, 0);
    Node* goal = new Node(0, 2);
    grid_map.grid[0][0] = 1;  // 设置障碍物
    grid_map.grid[0][1] = 1;  // 设置障碍物
    grid_map.grid[2][2] = 1;  // 设置障碍物

    std::cout << "start: (" << start->x << ", " << start->y << ")" << std::endl;
    std::cout << "goal: (" << goal->x << ", " << goal->y << ")" << std::endl;
    std::vector<Node*> path = a_star(grid_map, start, goal);
    std::cout << "path size: " << path.size() << std::endl;

    visualize_path(path);  // 可视化路径和地图
    std::cout << "Path found!" << std::endl;
    return 0;
}
