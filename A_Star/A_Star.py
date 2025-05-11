import heapq
import matplotlib.pyplot as plt
import numpy as np

# 地图
grid = np.array([[1, 1, 0], [0, 0, 0], [0, 0, 1]])

start = (2, 0)
goal = (0, 2)
rows, cols = grid.shape


# 启发函数：曼哈顿距离
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


# 邻居节点
def neighbors(pos):
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    result = []
    for d in dirs:
        nr, nc = pos[0] + d[0], pos[1] + d[1]
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0:
            result.append((nr, nc))
    return result


# A* 算法
def a_star(start, goal):
    open_set = []
    heapq.heappush(open_set, (heuristic(start, goal), 0, start))
    came_from = {}
    cost_so_far = {start: 0}

    while open_set:
        _, cost, current = heapq.heappop(open_set)

        if current == goal:
            path = []
            while current != start:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path

        for next in neighbors(current):
            new_cost = cost_so_far[current] + 1
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost
                priority = new_cost + heuristic(next, goal)
                heapq.heappush(open_set, (priority, new_cost, next))
                came_from[next] = current

    return None


# 求路径
path = a_star(start, goal)
print("路径:", path)

# 可视化
fig, ax = plt.subplots()
ax.imshow(grid, cmap="Greys", origin="upper")

# 画网格
ax.set_xticks(np.arange(-0.5, cols, 1))
ax.set_yticks(np.arange(-0.5, rows, 1))
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.grid(color="black")

# 路径
if path:
    path_x = [p[1] for p in path]
    path_y = [p[0] for p in path]
    ax.plot(path_x, path_y, "ro-", linewidth=2)
    ax.text(
        start[1], start[0], "S", color="blue", ha="center", va="center", fontsize=14
    )
    ax.text(goal[1], goal[0], "G", color="green", ha="center", va="center", fontsize=14)

plt.title("A* Pathfinding")
plt.show()