#include "Dijkstra.h"

double Dijkstra(const AdjacencyMatrix& adjacency_matrix, const int& start_vertex,
                const int& goal_vertex) {
    int n = adjacency_matrix.vertex_number;

    std::vector<VertexInfo> vertex_info(n, {std::numeric_limits<double>::infinity(), -1, false});
    vertex_info[start_vertex].distance = 0.0;

    using P = std::pair<double, int>;
    std::priority_queue<P, std::vector<P>, std::greater<P>> open_list;
    open_list.push({0.0, start_vertex});

    while (!open_list.empty()) {
        auto [dist_u, u] = open_list.top();
        open_list.pop();

        if (vertex_info[u].visited) continue;
        vertex_info[u].visited = true;

        if (u == goal_vertex) break;

        for (int i = 0; i < n; i++) {
            double weight = adjacency_matrix.adjacency_matrix_[u][i];
            if (weight > 0 && !vertex_info[i].visited) {
                double new_dist = dist_u + weight;
                if (new_dist < vertex_info[i].distance) {
                    vertex_info[i].distance = new_dist;
                    vertex_info[i].predecessor = u;
                    open_list.push({new_dist, i});
                }
            }
        }
    }
    return vertex_info[goal_vertex].distance;
}