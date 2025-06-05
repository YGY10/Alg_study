#include <iostream>
#include "A_Star.h"
#include "Dijkstra.h"

int main() {
    // A*算法示例
    // GridMap grid_map(3, 3);
    // Node* start = &grid_map.nodes_[0][2];  // 起点
    // Node* goal = &grid_map.nodes_[2][0];   // 终点
    // grid_map.nodes_[0][0].grid_value = 1;  // 设置障碍物
    // grid_map.nodes_[0][1].grid_value = 1;  // 设置障碍物
    // grid_map.nodes_[2][2].grid_value = 1;  // 设置障碍物
    // grid_map.nodes_[2][1].grid_value = 1;  // 设置障碍物

    // std::cout << "start: (" << start->x << ", " << start->y << ")" << std::endl;
    // std::cout << "goal: (" << goal->x << ", " << goal->y << ")" << std::endl;
    // std::vector<Node*> path = a_star(grid_map, start, goal);
    // std::cout << "path size: " << path.size() << std::endl;

    // visualize_path(path);  // 可视化路径和地图
    // std::cout << "Path found!" << std::endl;

    // Dijkstra算法示例
    AdjacencyMatrix adjacency_matrix;
    std::cout << "Please input the number of vertices and edges: ";
    std::cin >> adjacency_matrix.vertex_number >> adjacency_matrix.edge_number;
    adjacency_matrix.adjacency_matrix_.resize(
        adjacency_matrix.vertex_number, std::vector<double>(adjacency_matrix.vertex_number, 0.f));
    std::cout << "Please input the edges in the format: vertex1 vertex2 weight" << std::endl;
    for (int i = 0; i < adjacency_matrix.edge_number; ++i) {
        int vertex1, vertex2;
        double weight;
        std::cin >> vertex1 >> vertex2 >> weight;
        if (vertex1 < 0 || vertex1 >= adjacency_matrix.vertex_number || vertex2 < 0 ||
            vertex2 >= adjacency_matrix.vertex_number) {
            std::cerr << "Invalid vertex index: " << vertex1 << " or " << vertex2 << std::endl;
            return 1;
        }
        adjacency_matrix.adjacency_matrix_[vertex1][vertex2] = weight;
        adjacency_matrix.adjacency_matrix_[vertex2][vertex1] = weight;  // Assuming undirected graph
    }
    std::cout << "The adjacency matrix is:" << std::endl;
    for (const auto& row : adjacency_matrix.adjacency_matrix_) {
        for (const auto& value : row) {
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
    int start_vertex, goal_vertex;
    std::cout << "Please input the start vertex and goal vertex: ";
    std::cin >> start_vertex >> goal_vertex;
    double min_cost = Dijkstra(adjacency_matrix, start_vertex, goal_vertex);
    std::cout << "The minimum cost from vertex " << start_vertex << " to vertex " << goal_vertex
              << " is: " << min_cost << std::endl;
    return 0;
}
