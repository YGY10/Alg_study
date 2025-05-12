#include "A_Star.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <map>
#include <queue>

Node::Node(int x, int y)
    : x(x), y(y), g(INFINITY), h(0), f(INFINITY), parent(nullptr), grid_value(0) {}

bool Node::operator<(const Node& other) const {
    return f > other.f;  // 优先队列是最小堆，f值越小优先级越高
}

GridMap::GridMap(int w, int h) : width_(w), height_(h) {
    nodes_.resize(h, std::vector<Node>(w, Node(0, 0)));
    for (int i = 0; i < h; ++i) {
        for (int j = 0; j < w; ++j) {
            nodes_[i][j] = Node(i, j);  // 初始化节点坐标
        }
    }
}

bool GridMap::is_valid(int x, int y) {
    return x >= 0 && x < width_ && y >= 0 && y < height_ && nodes_[x][y].grid_value != 1;
}

std::vector<Node*> GridMap::get_neighbors(Node* current) {
    std::vector<Node*> neighbors;
    int dx[] = {-1, 1, 0, 0};  // 上下左右
    int dy[] = {0, 0, -1, 1};

    for (int i = 0; i < 4; ++i) {
        int nx = current->x + dx[i];
        int ny = current->y + dy[i];
        if (is_valid(nx, ny)) {
            neighbors.push_back(new Node(nx, ny));
        }
    }
    return neighbors;
}

float heuristic(Node* a, Node* b) {
    return abs(a->x - b->x) + abs(a->y - b->y);  // 曼哈顿距离
}

std::vector<Node*> a_star(GridMap& grid, Node* start, Node* goal) {
    std::priority_queue<Node*> open_list;
    std::map<std::pair<int, int>, Node*> all_nodes;
    std::map<std::pair<int, int>, bool> closed_list;  // 用于标记已访问的节点

    start->g = 0;
    start->f = heuristic(start, goal);
    open_list.push(start);
    all_nodes[{start->x, start->y}] = start;

    while (!open_list.empty()) {
        Node* current = open_list.top();
        open_list.pop();

        // 如果当前节点是目标节点，重建路径
        if (current->x == goal->x && current->y == goal->y) {
            std::vector<Node*> path;
            while (current != nullptr) {
                path.push_back(current);
                current = current->parent;
            }
            std::reverse(path.begin(), path.end());
            return path;
        }

        // 将当前节点添加到已访问列表
        closed_list[{current->x, current->y}] = true;

        std::vector<Node*> neighbors = grid.get_neighbors(current);
        for (Node* neighbor : neighbors) {
            if (closed_list.find({neighbor->x, neighbor->y}) != closed_list.end()) {
                continue;  // 如果已经访问过，跳过该节点
            }

            float tentative_g = current->g + 1.f;  // 假设每一步的成本为1

            // 只更新邻居，如果找到了更短的路径
            if (tentative_g < neighbor->g || neighbor->g == INFINITY) {
                neighbor->parent = current;
                neighbor->g = tentative_g;
                neighbor->h = heuristic(neighbor, goal);
                neighbor->f = neighbor->g + neighbor->h;

                // 如果邻居不在 open_list 中，加入它
                if (all_nodes.find({neighbor->x, neighbor->y}) == all_nodes.end()) {
                    open_list.push(neighbor);
                    all_nodes[{neighbor->x, neighbor->y}] = neighbor;
                }
            }
        }
    }
    return {};  // 如果没有路径返回空
}

void visualize_path(std::vector<Node*> path) {
    for (Node* node : path) {
        std::cout << "(" << node->x << ", " << node->y << ") ";
    }
    std::cout << std::endl;
}
