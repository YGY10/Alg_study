#ifndef A_STAR_H
#define A_STAR_H

#include <cmath>
#include <map>
#include <queue>
#include <vector>

class Node {
   public:
    int x, y;
    float g, h, f;  // g: 从起点到该节点的实际成本，h: 启发式成本，f: 总成本
    Node* parent;

    Node(int x, int y);
    bool operator<(const Node& other) const;
};

class GridMap {
   public:
    int width, height;
    std::vector<std::vector<int>> grid;

    GridMap(int w, int h);
    bool is_valid(int x, int y);
    std::vector<Node*> get_neighbors(Node* current);
};

float heuristic(Node* a, Node* b);
std::vector<Node*> a_star(GridMap& map, Node* start, Node* goal);
void visualize_path(std::vector<Node*> path);

#endif  // A_STAR_H
