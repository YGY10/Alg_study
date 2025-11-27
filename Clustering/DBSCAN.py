import numpy as np
import matplotlib.pyplot as plt
import enum
import sys
from typing import Dict, Callable

# ---------------- 定义枚举和类 ----------------


class PointType(enum.Enum):
    kpointType_UNDO = 0
    kpointType_CORE = 3
    kpointType_BORDER = 2
    kpointType_NOISE = 1


class ParticleType(enum.Enum):
    kparticleType_UNKNOWN = 0
    kparticleType_LaneLine = 3


class keyVal(enum.Enum):
    keyVal_Y = 2      # y距离
    keyVal_CENTER = 3  # 欧氏距离


class Particle:
    def __init__(self, id, x, y, weight=1.0, type=ParticleType.kparticleType_UNKNOWN):
        self.obj_id = id
        self.x = x
        self.y = y
        self.weight = weight
        self.particle_type = type
        self.cluster = 0
        self.point_type = PointType.kpointType_UNDO
        self.pts = 0
        self.corePointID = -1
        self.corepts = []   # 保存 dataset 的索引
        self.visited = 0
        self.cluster_idx = 0


def squareDistance(a, b) -> float:
    return (a.x - b.x)**2 + (a.y - b.y)**2


def CalcuYDistance(a, b) -> float:
    threshold = 50.0
    if abs(a.x - b.x) > threshold:
        return sys.float_info.max
    return abs(a.y - b.y)

# ---------------- DBSCAN 算法 ----------------


def DBSCAN(dataset, KeyValType, eps, minPts):
    clusterID = 0
    length = len(dataset)
    # 没数据直接返回
    if length == 0:
        return
    # 选距离函数
    dist_func_map: Dict[keyVal, Callable[[Particle, Particle], float]] = {
        keyVal.keyVal_CENTER: squareDistance,
        keyVal.keyVal_Y: CalcuYDistance
    }
    dist_func = dist_func_map[KeyValType]

    dist_p2p = np.zeros((length, length), dtype=np.float32)
    core_point = []  # 存索引

    # Step1: 找核心点
    for i in range(length):
        dataset[i].pts = 1
        for j in range(i+1, length):
            distance = dist_func(dataset[i], dataset[j])
            dist_p2p[i][j] = distance
            dist_p2p[j][i] = distance
            if distance <= eps:
                # 小于阈值，认为此点的邻居点+1
                dataset[i].pts += 1
                dataset[j].pts += 1
        # 邻居点的个数大于阈值，认为此点为核心点
        if dataset[i].pts >= minPts:
            dataset[i].point_type = PointType.kpointType_CORE
            dataset[i].corePointID = i
            core_point.append(i)

    # Step2: 建立核心点邻居关系
    # 遍历核心点
    for i in range(len(core_point)):
        dist_i = dist_p2p[core_point[i]]
        corepts_i = dataset[core_point[i]].corepts
        for j in range(len(core_point)):
            distTemp = dist_i[core_point[j]]
            # 两个核心点的距离小于阈值
            # 将一个核心点放在另一个核心点的corepts中
            if distTemp <= eps:
                corepts_i.append(core_point[j])  # 存 dataset 索引

    # Step3: 扩展簇
    # 遍历核心点
    for i in range(len(core_point)):
        ps = []
        # 跳过已经访问过的核心点
        if dataset[core_point[i]].visited == 1:
            continue
        # 给没访问过的核心点一个簇ID
        clusterID += 1
        dataset[core_point[i]].cluster = clusterID
        ps.append(dataset[core_point[i]])
        while ps:
            # 标记此核心点为访问过
            v = ps.pop()
            v.visited = 1
            # 遍历此核心点的邻居核心点
            for idx in v.corepts:
                # 跳过已经访问过的核心点
                if dataset[idx].visited == 1:
                    continue
                # 赋予同样的簇ID，标记为访问过
                dataset[idx].cluster = clusterID
                dataset[idx].visited = 1
                ps.append(dataset[idx])

    # Step4: 标记边界点
    # 遍历所有点
    for i in range(length):
        # 如果这个点是核心点，跳过，因为核心点在第三步已经处理过了
        if dataset[i].point_type == PointType.kpointType_CORE:
            continue
        for j in range(len(core_point)):
            # distTemp为此点与第j个核心点的距离
            distTemp = dist_p2p[i][core_point[j]]
            # 如果小于阈值，认为此点为边界点（也就是这个点落在某个核心点的邻域内， 但是他本身没有足够多的邻域点）
            if distTemp <= eps:
                dataset[i].point_type = PointType.kpointType_BORDER
                dataset[i].cluster = dataset[core_point[j]].cluster
                break


# ---------------- 测试部分 ----------------
if __name__ == "__main__":
    np.random.seed(0)

    # 构造三簇随机点
    data1 = np.random.randn(30, 2) * 0.8 + np.array([-1, 4])
    data2 = np.random.randn(30, 2) * 0.3 + np.array([1, 1])
    data3 = np.random.randn(30, 2) * 0.3 + np.array([2, 3])
    data = np.vstack((data1, data2, data3))

    # 转换为 Particle
    dataset = [Particle(str(i), x, y) for i, (x, y) in enumerate(data)]

    # 运行 DBSCAN
    DBSCAN(dataset, keyVal.keyVal_CENTER, eps=0.50, minPts=4)

    # 画图
    plt.figure(figsize=(6, 6))

    for cluster_id in set(p.cluster for p in dataset):
        # 取该簇所有点
        cluster_points = [p for p in dataset if p.cluster == cluster_id]

        # 分开核心点和边界点
        core_x = [p.x for p in cluster_points if p.point_type ==
                  PointType.kpointType_CORE]
        core_y = [p.y for p in cluster_points if p.point_type ==
                  PointType.kpointType_CORE]
        border_x = [p.x for p in cluster_points if p.point_type ==
                    PointType.kpointType_BORDER]
        border_y = [p.y for p in cluster_points if p.point_type ==
                    PointType.kpointType_BORDER]

        # 用 cluster_id 控制颜色
        color = plt.cm.tab10(cluster_id % 10)

        # 核心点：大圆点
        plt.scatter(core_x, core_y, c=[
                    color], marker="o", s=80, label=f"Cluster {cluster_id} (core)")
        # 边界点：小圆点
        plt.scatter(border_x, border_y, c=[
                    color], marker=".", s=40, label=f"Cluster {cluster_id} (border)")

    # 画噪声点
    noise_x = [p.x for p in dataset if p.point_type ==
               PointType.kpointType_NOISE]
    noise_y = [p.y for p in dataset if p.point_type ==
               PointType.kpointType_NOISE]
    if noise_x:
        plt.scatter(noise_x, noise_y, c="black",
                    marker="x", s=60, label="Noise")

    plt.title("DBSCAN Clustering Result")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()
    plt.show()
