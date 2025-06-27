import re
import matplotlib.pyplot as plt

conf_known_sample = []
conf_novel_sample = []


with open("./outputs/dataset_cifar100_lbl_percent_50_novel_percent_50_27-06-25_1157_exp3/score_logger_base.txt", "r") as f:
    for line in f:
        if "Novel Detection ACC:" in line:
            match = re.search(r"Novel Detection ACC:\s*([\d.]+)", line)
            if match:
                conf_known_sample.append(float(match.group(1)))
        if len(conf_known_sample) == 195:
            break
           
with open("./outputs/dataset_cifar100_lbl_percent_50_novel_percent_50_27-06-25_1530_exp4/score_logger_base.txt", "r") as f:
    for line in f:
        if "Novel Detection ACC:" in line:
            match = re.search(r"Novel Detection ACC:\s*([\d.]+)", line)
            if match:
                conf_novel_sample.append(float(match.group(1)))
        if len(conf_novel_sample) == 200:
            break           
            
            

iters = list(range(len(conf_known_sample)))
sample_count1 = conf_known_sample
sample_count2 = conf_novel_sample

plt.figure(figsize=(6, 4))

# 绘制四条折线
plt.plot(iters, sample_count1, label='coop with labeled sample', marker='o', markersize=4)
plt.plot(iters, sample_count2, label='clip zero shot', marker='s', markersize=4)
# 在y=0.9134处画一条横线
# plt.axhline(y=0.9134, color='r', linestyle='--', label='clip zero shot')

# 添加标签和图例
plt.xlabel('iter')
plt.ylabel('ACC')
plt.title('Novel Sample Detection ACC')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.savefig('draw/novel_sample_detection_acc.png', dpi=300)
plt.show()
