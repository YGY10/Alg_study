import torch


class NNPlanner:
    def __init__(self, model=None):
        self.model = model  # 先 None

    def plan(self, ego, obstacles, goal):
        """
        返回一条局部轨迹（占位）
        """
        # TODO: 后面换成神经网络
        path = []
        for i in range(1, 20):
            path.append((ego.x + i * 1.0, ego.y))
        return path
