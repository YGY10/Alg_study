import csv

CSV_PATH = "Simulation/nn_dataset_with_epi.csv"

behind_cnt = 0
behind_l_vals = []

far_cnt = 0
far_l_vals = []

FAR_THRESH = 10.0  # 你可以根据自己设定改

with open(CSV_PATH, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        if not row:
            continue

        # 列结构：
        # 0 episode_id
        # 1 step_id
        # 2 ego.v
        # 3 ego.l
        # 4 ego.dl
        # 5 ego.ddl
        # 6 ref.yaw
        # 7 obs1.dx
        # 8 obs1.dy
        # 9 obs1.v
        # 10 obs2.dx
        # 11 obs2.dy
        # 12 obs2.v
        # 13 label_l

        dx1 = float(row[-5])
        dx2 = float(row[-8])
        l_label = float(row[-2])

        # 情况1：两个障碍都在车后面
        if dx1 < 0 and dx2 < 0:
            behind_cnt += 1
            behind_l_vals.append(l_label)

        # 情况2：两个障碍都离得很远（在前方）
        if dx1 > FAR_THRESH and dx2 > FAR_THRESH:
            far_cnt += 1
            far_l_vals.append(l_label)

print("=== 两个障碍都在车后面（dx<0） ===")
print("样本数:", behind_cnt)
if behind_cnt > 0:
    print("label_l 示例:", behind_l_vals[:])

print("\n=== 两个障碍都很远（dx>FAR_THRESH） ===")
print("样本数:", far_cnt)
if far_cnt > 0:
    print("label_l 示例:", far_l_vals[:])
