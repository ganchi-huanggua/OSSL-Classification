import re
import matplotlib.pyplot as plt

logits_known_0_50 = []
logits_known_50_100 = []
logits_novel_0_50 = []
logits_novel_50_100 = []

with open("./outputs/dataset_cifar100_lbl_percent_50_novel_percent_50_26-06-25_1631_exp2/score_logger_base.txt", "r") as f:
    for line in f:
        if "logits_dist_known_0-50" in line:
            match = re.search(r"logits_dist_known_0-50:\s*([\d.]+)", line)
            if match:
                logits_known_0_50.append(float(match.group(1)))
        elif "logits_dist_known_50-100" in line:
            match = re.search(r"logits_dist_known_50-100:\s*([\d.]+)", line)
            if match:
                logits_known_50_100.append(float(match.group(1)))
        elif "logits_dist_novel_0-50" in line:
            match = re.search(r"logits_dist_novel_0-50:\s*([\d.]+)", line)
            if match:
                logits_novel_0_50.append(float(match.group(1)))
        elif "logits_dist_novel_50-100" in line:
            match = re.search(r"logits_dist_novel_50-100:\s*([\d.]+)", line)
            if match:
                logits_novel_50_100.append(float(match.group(1)))
            

iters = list(range(len(logits_known_0_50)))
sample_count1 = logits_known_0_50
sample_count2 = logits_known_50_100
sample_count3 = logits_novel_0_50
sample_count4 = logits_novel_50_100

plt.figure(figsize=(6, 4))

# 绘制四条折线
plt.plot(iters, sample_count1, label='known sample logits: class 0-50', marker='o', markersize=4)
plt.plot(iters, sample_count2, label='known sample logits: class 50-100', marker='s', markersize=4)
plt.plot(iters, sample_count3, label='novel sample logits: class 0-50', marker='^', markersize=4)
plt.plot(iters, sample_count4, label='novel sample logits: class 50-100', marker='v', markersize=4)

# 添加标签和图例
plt.xlabel('iter')
plt.ylabel('logits')
plt.title('logits over Iterations')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.savefig('draw/logits_plot.png', dpi=300)
plt.show()
