import numpy as np
import math
from numpy.linalg import inv, norm
from scipy.interpolate import CubicSpline


def CalculateCurvatureDefault(xy_points):
    points_size = len(xy_points)
    # 计算弧长
    accumulated_s = [0.0] * points_size
    distance = 0.0

    fx, fy = xy_points[0]
    for i in range(1, points_size):
        nx, ny = xy_points[i]
        seg = math.sqrt((fx - nx)**2 + (fy - ny)**2)
        accumulated_s[i] = seg + distance
        distance += seg
        fx, fy = nx, ny

    # 计算xds, yds
    xds_list = [0.0] * points_size
    yds_list = [0.0] * points_size

    xdds_list = [0.0] * points_size
    ydds_list = [0.0] * points_size
    for i in range(points_size):
        if i == 0:
            xds = (xy_points[i+1][0] - xy_points[i][0]) / (accumulated_s[i+1] - accumulated_s[i])
            yds = (xy_points[i+1][1] - xy_points[i][1]) / (accumulated_s[i+1] - accumulated_s[i])
        elif i == points_size - 1:
            xds = (xy_points[i][0] - xy_points[i-1][0]) / (accumulated_s[i] - accumulated_s[i-1])
            yds = (xy_points[i][1] - xy_points[i-1][1]) / (accumulated_s[i] - accumulated_s[i-1])
        else:
            xds = (xy_points[i+1][0] - xy_points[i-1][0]) / (accumulated_s[i+1] - accumulated_s[i-1])
            yds = (xy_points[i+1][1] - xy_points[i-1][1]) / (accumulated_s[i+1] - accumulated_s[i-1])

        xds_list[i] = xds
        yds_list[i] = yds
    # 计算xdds, ydds
    for i in range(points_size):
        if i == 0:
            xdds = (xds_list[i+1] - xds_list[i]) / (accumulated_s[i+1] - accumulated_s[i])
            ydds = (yds_list[i+1] - yds_list[i]) / (accumulated_s[i+1] - accumulated_s[i])
        elif i == points_size - 1:
            xdds = (xds_list[i] - xds_list[i-1]) / (accumulated_s[i] - accumulated_s[i-1])
            ydds = (yds_list[i] - yds_list[i-1]) / (accumulated_s[i] - accumulated_s[i-1])
        else:
            xdds = (xds_list[i+1] - xds_list[i-1]) / (accumulated_s[i+1] - accumulated_s[i-1])
            ydds = (yds_list[i+1] - yds_list[i-1]) / (accumulated_s[i+1] - accumulated_s[i-1])

        xdds_list[i] = xdds
        ydds_list[i] = ydds
    # 计算曲率
    kappas = [0.0] * points_size

    for i in range(points_size):
        xds = xds_list[i]
        yds = yds_list[i]
        xdds = xdds_list[i]
        ydds = ydds_list[i]

        denom = math.sqrt(xds*xds + yds*yds) * (xds*xds + yds*yds) + 1e-6

        kappa = (xds * ydds - yds * xdds) / denom
        kappas[i] = kappa
    return kappas

import numpy as np
import math

def CalculateCurvatureDifference(xy_points, interval=1):
    data = np.array(xy_points)
    N = len(data)

    # 计算弧长 s
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i-1] + math.hypot(data[i][0] - data[i-1][0],
                                   data[i][1] - data[i-1][1])

    kappa = np.zeros(N)

    for i in range(N):
        # 边缘点处理
        if i < 2*interval or i > N - 2*interval - 1:
            continue

        # 取五个点：i-2, i-1, i, i+1, i+2
        idx = [i-2*interval, i-interval, i, i+interval, i+2*interval]

        xs = data[idx, 0]
        ys = data[idx, 1]
        ss = s[idx]

        # 对 s 求一阶、二阶导数 (用中心差分)
        dxds = np.gradient(xs, ss)
        dyds = np.gradient(ys, ss)

        d2xds2 = np.gradient(dxds, ss)
        d2yds2 = np.gradient(dyds, ss)

        # 使用中间点的值（即本地最准确的那个）
        x1 = dxds[2]
        y1 = dyds[2]
        x2 = d2xds2[2]
        y2 = d2yds2[2]

        denom = math.sqrt(x1*x1 + y1*y1) * (x1*x1 + y1*y1) + 1e-6
        kappa[i] = (x1*y2 - y1*x2) / denom

    # 边缘平滑：把两端缺失的点复制邻域值
    kappa[:2*interval] = kappa[2*interval]
    kappa[-2*interval:] = kappa[-2*interval-1]

    return kappa.tolist()



def CalculateCurvatureParameter(xy_points, interval=1):
    data = np.array(xy_points)
    N = len(data)

    # 计算弧长 s
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i-1] + math.hypot(data[i][0]-data[i-1][0],
                                   data[i][1]-data[i-1][1])

    kappa = np.zeros(N)

    for i in range(N):
        if i < interval or i > N - interval - 1:
            continue

        idx = [i-interval, i, i+interval]

        xs = data[idx, 0]
        ys = data[idx, 1]
        ss = s[idx]

        # 构造二次曲线 ax^2 + bx + c   （对 s）
        M = np.array([
            [ss[0]**2, ss[0], 1],
            [ss[1]**2, ss[1], 1],
            [ss[2]**2, ss[2], 1]
        ])

        M += np.eye(3)*1e-6

        ax, bx, cx = np.linalg.solve(M, xs)
        ay, by, cy = np.linalg.solve(M, ys)

        # 一阶导 dx/ds, dy/ds
        x1 = 2*ax*ss[1] + bx
        y1 = 2*ay*ss[1] + by

        # 二阶导 d²x/ds², d²y/ds²
        x2 = 2*ax
        y2 = 2*ay

        denom = math.sqrt(x1*x1 + y1*y1) * (x1*x1 + y1*y1) + 1e-6
        kappa[i] = (x1*y2 - y1*x2) / denom

    # 边缘点补齐
    kappa[:interval] = kappa[interval]
    kappa[-interval:] = kappa[-interval-1]

    return kappa.tolist()



def CalculateCurvatureCircle(xy_points, interval=1):
    data = np.array(xy_points)
    N = len(data)

    # 计算 s
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i-1] + math.hypot(data[i][0]-data[i-1][0],
                                   data[i][1]-data[i-1][1])

    # 每隔 interval 米选个点比较复杂，因此这里保持 index，但曲率公式不变
    # 若你想实现“按 s 找邻点”，我可以帮你写
    # 但保持 index 更常见、也更稳定

    kappa = np.zeros(N)

    for i in range(N):
        if i < interval or i > N - interval - 1:
            continue

        idx = [i-interval, i, i+interval]
        x1, x2, x3 = data[idx,0]
        y1, y2, y3 = data[idx,1]

        denom = 2*(x1*(y2-y3)+x2*(y3-y1)+x3*(y1-y2))

        xc = ((x1**2+y1**2)*(y2-y3) +
              (x2**2+y2**2)*(y3-y1) +
              (x3**2+y3**2)*(y1-y2)) / denom

        yc = ((x1**2+y1**2)*(x3-x2) +
              (x2**2+y2**2)*(x1-x3) +
              (x3**2+y3**2)*(x2-x1)) / denom

        r = math.sqrt((x1-xc)**2 + (y1-yc)**2)
        k = 1/r

        # 左右方向判断
        cross = (x2-x1)*(y3-y2) - (y2-y1)*(x3-x2)
        if cross < 0:
            k = -k

        kappa[i] = k

    kappa[:interval] = kappa[interval]
    kappa[-interval:] = kappa[-interval-1]

    return kappa.tolist()




from scipy.interpolate import CubicSpline

def CalculateCurvatureSpline(xy_points):
    data = np.array(xy_points)
    N = len(data)

    # 计算弧长 s
    s = np.zeros(N)
    for i in range(1, N):
        s[i] = s[i-1] + math.hypot(data[i][0]-data[i-1][0],
                                   data[i][1]-data[i-1][1])

    x = data[:,0]
    y = data[:,1]

    # 用s作为插值自变量
    cs_x = CubicSpline(s, x)
    cs_y = CubicSpline(s, y)

    x1 = cs_x(s, 1)
    y1 = cs_y(s, 1)

    x2 = cs_x(s, 2)
    y2 = cs_y(s, 2)

    denom = np.sqrt(x1*x1 + y1*y1) * (x1*x1 + y1*y1) + 1e-6
    kappa = (x1*y2 - y1*x2) / denom

    return kappa.tolist()

