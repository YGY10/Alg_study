import math
import matplotlib.pyplot as plt
import CalculateCurvature


def ComputePathProfile(xy_points):

    # 输入检查
    if len(xy_points) < 3:
        print("[ComputePathProfile] input xy points size < 3")
        return False, [], [], [], []

    # =======================
    # Step 1: 初始化容器
    # =======================
    points_size = len(xy_points)

    dxs = [0.0] * points_size
    dys = [0.0] * points_size

    xds_list = [0.0] * points_size
    yds_list = [0.0] * points_size

    xdds_list = [0.0] * points_size
    ydds_list = [0.0] * points_size

    headings = []
    kappas_default = []
    kappas_diff = []
    kappas_param = []
    kappas_circle = []
    kappas_spline = []

    dkappas = []

    # =======================
    # Step 2: 计算 dx 和 dy（有限差分）
    # =======================
    for i in range(points_size):
        if i == 0:
            x_delta = xy_points[i+1][0] - xy_points[i][0]
            y_delta = xy_points[i+1][1] - xy_points[i][1]
        elif i == points_size - 1:
            x_delta = xy_points[i][0] - xy_points[i-1][0]
            y_delta = xy_points[i][1] - xy_points[i-1][1]
        else:
            x_delta = 0.5 * (xy_points[i+1][0] - xy_points[i-1][0])
            y_delta = 0.5 * (xy_points[i+1][1] - xy_points[i-1][1])

        dxs[i] = x_delta
        dys[i] = y_delta

    # =======================
    # Step 3: 计算 heading = atan2(dy, dx)
    # =======================
    for i in range(points_size):
        heading = math.degrees(math.atan2(dys[i], dxs[i]))
        headings.append(heading)

    kappas_default = CalculateCurvature.CalculateCurvatureDefault(xy_points)
    kappas_diff = CalculateCurvature.CalculateCurvatureDifference(xy_points)
    kappas_param = CalculateCurvature.CalculateCurvatureParameter(xy_points)
    kappas_spline = CalculateCurvature.CalculateCurvatureSpline(xy_points)
    kappas_circle = CalculateCurvature.CalculateCurvatureCircle(xy_points)

    accumulated_s = [0.0] * points_size
    distance = 0.0

    fx, fy = xy_points[0]
    for i in range(1, points_size):
        nx, ny = xy_points[i]
        seg = math.sqrt((fx - nx)**2 + (fy - ny)**2)
        accumulated_s[i] = seg + distance
        distance += seg
        fx, fy = nx, ny

    # =======================
    # Step 8: 求 dkappa
    # =======================
    dkappas = [0.0] * points_size

    
    return True, headings, accumulated_s, kappas_default, kappas_diff, kappas_param, kappas_circle, kappas_spline, dkappas


def ExtendLineWithCurve(xy_points, headings, kappas, max_front_s, step):

    # --- 取末端 yaw（平均末端 5m）
    end_s = 0.0
    for i in range(1, len(xy_points)):
        x1, y1 = xy_points[i-1]
        x2, y2 = xy_points[i]
        end_s += math.hypot(x2-x1, y2-y1)

    # 最后 5m 内的 yaw 平均
    yaw_sum = 0
    count = 0
    acc_s = [0.0]
    for i in range(1, len(xy_points)):
        ds = math.hypot(xy_points[i][0]-xy_points[i-1][0],
                        xy_points[i][1]-xy_points[i-1][1])
        acc_s.append(acc_s[-1] + ds)

    for i in range(len(acc_s)-1, -1, -1):
        if acc_s[-1] - acc_s[i] > 5.0:
            break
        yaw_sum += math.radians(headings[i])
        count += 1

    if count == 0:
        return xy_points

    extend_yaw = yaw_sum / count

    # --- 末端曲率：取最后 30m 的点，然后取中位数，再除 2
    k_list = []
    for i in range(len(acc_s)-1, -1, -1):
        if acc_s[-1] - acc_s[i] > 30.0:
            break
        k_list.append(kappas[i])

    if len(k_list) == 0:
        return xy_points

    k_list.sort()
    extend_kappa = k_list[len(k_list)//2] / 2
    print("extend kappa", extend_kappa)
    # 限制范围
    extend_kappa = max(min(extend_kappa, 0.005), -0.005)

    # --- 进行延长 ---
    new_points = list(xy_points)
    last_x, last_y = xy_points[-1]

    last_s = acc_s[-1]

    while last_s < max_front_s:
        new_x = last_x + step
        dx = new_x - xy_points[-1][0]

        new_y = xy_points[-1][1] + extend_yaw * dx + extend_kappa * dx * dx

        new_points.append((new_x, new_y))

        last_s += math.hypot(new_x - last_x, new_y - last_y)
        last_x, last_y = new_x, new_y

    return new_points



def print_all_kappa_table(xy_points, headings,
                          kappa_default, kappa_diff,
                          kappa_param, kappa_circle,
                          kappa_spline):

    print(" i |      x       |      y       | heading_deg |   k_default    |    k_diff      |   k_param      |   k_circle     |   k_spline")
    print("-" * 140)

    N = len(xy_points)
    for i in range(N):
        x, y = xy_points[i]
        print(f"{i:2d} | {x:11.6f} | {y:11.6f} | {headings[i]:11.6f} | "
              f"{kappa_default[i]:13.8f} | {kappa_diff[i]:13.8f} | "
              f"{kappa_param[i]:13.8f} | {kappa_circle[i]:13.8f} | {kappa_spline[i]:13.8f}")




xy_points = [
    (-59.2415, 2.19727),
    (-57.7871, 2.17464),
    (-54.6941, 2.16275),
    (-51.6909, 2.17550),
    (-48.6872, 2.19928),
    (-45.6846, 2.19893),
    (-42.6819, 2.19804),
    (-39.6799, 2.17978),
    (-36.6807, 2.09976),
    (-33.6593, 2.03463),

    (-30.6550, 1.85486),
    (-27.6550, 1.28533),
    (-24.6550, 0.923111),
    (-21.6550, 0.647298),
    (-18.6550, 0.414554),
    (-15.6550, 0.198727),
    (-12.6550, -0.0506563),
    (-9.65501, -0.154760),
    (-6.65501, -0.188866),
    (-3.65501, -0.205608),

    (-0.655006, -0.214013),
    (2.19547, -0.278203),
    (5.16909, -0.235445),
    (8.26157, -0.161697),
    (9.85135, -0.110760),
    (11.4689, -0.0457082),
]


ok, headings, s, kappa_default, kappa_diff, kappa_param, kappa_circle, kappa_spline, dkappas = ComputePathProfile(xy_points)
max_s = s[-1] + 40.0

print_all_kappa_table(
    xy_points, headings,
    kappa_default, kappa_diff,
    kappa_param, kappa_circle,
    kappa_spline
)

ext_default = ExtendLineWithCurve(xy_points, headings, kappa_default, max_s, step=4)
ext_diff    = ExtendLineWithCurve(xy_points, headings, kappa_diff,    max_s, step=4)
ext_param   = ExtendLineWithCurve(xy_points, headings, kappa_param,   max_s, step=4)
ext_circle  = ExtendLineWithCurve(xy_points, headings, kappa_circle,  max_s, step=4)
ext_spline  = ExtendLineWithCurve(xy_points, headings, kappa_spline,  max_s, step=4)


plt.figure(figsize=(14,7))

# 原始轨迹
xs = [p[0] for p in xy_points]
ys = [p[1] for p in xy_points]
plt.plot(xs, ys, 'k.-', label="Original")

def draw(ext, name, color):
    xs = [p[0] for p in ext]
    ys = [p[1] for p in ext]
    plt.plot(xs, ys, color, label=name)

draw(ext_default, "Default", 'r-')
draw(ext_diff,    "Difference", 'g-')
draw(ext_param,   "Parameter", 'b-')
draw(ext_circle,  "Circle", 'm-')
draw(ext_spline,  "Spline", 'c-')

plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Path Extension Comparison (5 Curvature Methods)")
plt.grid(True)
plt.axis('equal')
plt.show()
