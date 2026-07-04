import csv
import math

IDs = []
MTs = []
hits = 0
total = 0

with open("fitts_log.csv") as f:
    reader = csv.DictReader(f)
    for row in reader:
        D = float(row["D"])
        W = float(row["W"])
        MT = float(row["MT"])
        hit = int(row["hit"])

        ID = math.log2(D / W + 1)

        IDs.append(ID)
        MTs.append(MT)
        hits += hit
        total += 1

TP = sum(IDs) / sum(MTs)
error_rate = (total - hits) / total * 100
avg_MT = sum(MTs) / len(MTs)

print("\n=== FITTS' LAW METRICS ===")
print(f"Throughput (TP): {TP:.2f} bits/s")
print(f"Average Movement Time: {avg_MT*1000:.1f} ms")
print(f"Error Rate: {error_rate:.2f} %")
print("=========================\n")
