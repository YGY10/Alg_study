import csv
import os
import tempfile
import shutil

INPUT_CSV = "./Simulation/nn_dataset_with_epi.csv"

STEPS_PER_EPISODE = 100  # 10s / 0.1s
KEEP_STEP_MOD = 5  # 0.5s 采样

# 用临时文件，避免一边读一边写同一个文件
fd, TMP_CSV = tempfile.mkstemp(suffix=".csv")
os.close(fd)

with open(INPUT_CSV, "r", newline="") as fin, open(TMP_CSV, "w", newline="") as fout:
    reader = csv.reader(fin)
    writer = csv.writer(fout)

    for row in reader:
        if not row:
            continue

        episode_id = int(row[0])
        step_id = int(row[1])

        # ===== 0.5s 降采样 =====
        if step_id % KEEP_STEP_MOD != 0:
            continue

        writer.writerow(row)

# 用处理后的文件覆盖原文件
shutil.move(TMP_CSV, INPUT_CSV)

print("done: downsampled in-place ->", INPUT_CSV)
