import numpy as np
import matplotlib.pyplot as plt
import math

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

class Line:
    def __init__(self,start_point:Point, end_point:Point):
        self.start_x = start_point.x
        self.start_y = start_point.y
        self.end_x = end_point.x
        self.end_y = end_point.y
        self.heading = math.atan2(self.end_y - self.start_y, self.end_x - self.start_x)
        self.length = math.sqrt((self.end_x - self.start_x)**2 + (self.end_y - self.start_y)**2)
        self.center = Point((self.start_x + self.end_x) / 2, (self.start_y + self.end_y) / 2)
    
    

def transform2LCS(line:Line, orign_point:Point) -> Point:
    transfor_mat = np.eye(3)
    transfor_mat[0, 0] = - math.sin(line.heading)
    transfor_mat[0, 1] = - math.cos(line.heading)
    transfor_mat[0, 2] = line.center.x
    transfor_mat[1, 0] = math.cos(line.heading) 
    transfor_mat[1, 1] = - math.sin(line.heading)
    transfor_mat[1, 2] = line.center.y
    transfor_mat = np.linalg.inv(transfor_mat)
    # 开始转化
    transfored_point = transfor_mat @ np.array([orign_point.x, orign_point.y, 1.0])
    return Point(transfored_point[0], transfored_point[1])


line = Line(Point(0, 0), Point(2, 0))
orign_point = Point(1, 1)
after_point = transform2LCS(line, orign_point)



# 一行两列
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 第一个图，画线：从起始点到终止点的箭头， 画转换前的点
axes[0].plot([line.start_x, line.end_x], [line.start_y, line.end_y], color='blue')
axes[0].arrow(line.start_x, line.start_y, line.end_x - line.start_x, line.end_y - line.start_y, color='blue', width=0.005)
axes[0].scatter(orign_point.x, orign_point.y, c='red', label="original")
axes[0].set_xlabel("x")
axes[0].set_ylabel("y")
axes[0].set_title("original")
axes[0].legend()
axes[0].grid(True)

# 第二个图，画转化后的点
axes[1].scatter(after_point.x, after_point.y, color='red', label="transformed")
axes[1].set_xlabel("x")
axes[1].set_ylabel("y")
axes[1].set_title("transformed")
axes[1].legend()
axes[1].grid(True)

plt.show()
