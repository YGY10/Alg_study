#include <cmath>
#include <iostream>
#include <limits>
#include <queue>
#include <vector>

struct AdjacencyMatrix {
    int vertex_number;  // 顶点数量
    int edge_number;    // 边数量
    std::vector<std::vector<double>> adjacency_matrix_;
};

struct VertexInfo {
    double distance;  // 从起点到该顶点的最短距离
    int predecessor;  // 前驱顶点的索引
    bool visited;     // 是否已访问
};

double Dijkstra(const AdjacencyMatrix& adjacency_matrix, const int& start_vertex,
                const int& goal_vertex);
