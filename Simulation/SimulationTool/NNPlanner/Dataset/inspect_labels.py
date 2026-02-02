import csv
from collections import Counter

csv_path = "nn_dataset.csv"

with open(csv_path, "r") as f:
    reader = csv.reader(f)
    raw_header = next(reader)

    header = [h.strip() for h in raw_header]
    print("Header (stripped):", header)

    danger_idx = header.index("target lane danger")
    l_idx = header.index("l target")

    cnt = Counter()
    l_counter = Counter()

    for row in reader:
        if not row:
            continue

        danger = int(float(row[danger_idx]))
        l = float(row[l_idx])

        cnt[danger] += 1
        if danger == 1:
            l_counter[round(l, 2)] += 1

total = cnt[0] + cnt[1]
print("\nDanger count:", cnt)
print("Danger ratio:", cnt[1] / total if total > 0 else 0.0)

print("\n[l_target distribution | danger=1]")
for k, v in l_counter.most_common(10):
    print(f"  l={k}: {v}")
