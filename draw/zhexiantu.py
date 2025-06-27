import re
import matplotlib.pyplot as plt

novel_total_list = []
pred_as_novel_list = []

with open("./outputs/dataset_cifar100_lbl_percent_50_novel_percent_50_26-06-25_1028_exp1/score_logger_base.txt", "r") as f:
    for line in f:
        match = re.search(r"Novel Total: (\d+), Pred as Novel: (\d+)", line)
        if match:
            novel_total = int(match.group(1))
            pred_as_novel = int(match.group(2))
            novel_total_list.append(novel_total)
            pred_as_novel_list.append(pred_as_novel)
            

iters = list(range(len(novel_total_list)))
sample_count1 = novel_total_list
sample_count2 = pred_as_novel_list

plt.figure(figsize=(6, 4))

# 绘制两条折线
plt.plot(iters, sample_count1, label='novel samples count', marker='o', markersize=4)
plt.plot(iters, sample_count2, label='pred novel samples count', marker='s', markersize=4)

# 添加标签和图例
plt.xlabel('iter')
plt.ylabel('Sample Count')
plt.title('Sample Count over Iterations')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.savefig('draw/sample_count_plot.png', dpi=300)
plt.show()
