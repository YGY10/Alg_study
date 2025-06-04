#include <iostream>
#include "A_Star.h"

int main() {
    GridMap grid_map(3, 3);
    Node* start = &grid_map.nodes_[0][2];  // 起点
    Node* goal = &grid_map.nodes_[2][0];   // 终点
    grid_map.nodes_[0][0].grid_value = 1;  // 设置障碍物
    grid_map.nodes_[0][1].grid_value = 1;  // 设置障碍物
    grid_map.nodes_[2][2].grid_value = 1;  // 设置障碍物
    grid_map.nodes_[2][1].grid_value = 1;  // 设置障碍物

    std::cout << "start: (" << start->x << ", " << start->y << ")" << std::endl;
    std::cout << "goal: (" << goal->x << ", " << goal->y << ")" << std::endl;
    std::vector<Node*> path = a_star(grid_map, start, goal);
    std::cout << "path size: " << path.size() << std::endl;

    visualize_path(path);  // 可视化路径和地图
    std::cout << "Path found!" << std::endl;
    return 0;
}
