import re
import matplotlib.pyplot as plt

conf_known_sample = []
conf_novel_sample = []


with open("./outputs/dataset_cifar100_lbl_percent_50_novel_percent_50_27-06-25_1530_exp4/score_logger_base.txt", "r") as f:
    for line in f:
        if "Known Class Discrepancy Energy:" in line:
            match = re.search(r"Known Class Discrepancy Energy:\s*([\d.]+)", line)
            if match:
                conf_known_sample.append(float(match.group(1)))
        elif "Novel Class Discrepancy Energy" in line:
            match = re.search(r"Novel Class Discrepancy Energy:\s*([\d.]+)", line)
            if match:
                conf_novel_sample.append(float(match.group(1)))
        if len(conf_novel_sample) == 200:
            break
            

iters = list(range(len(conf_novel_sample)))
sample_count1 = conf_known_sample
sample_count2 = conf_novel_sample

plt.figure(figsize=(6, 4))

# 绘制四条折线
plt.plot(iters, sample_count1, label='known sample ED ', marker='o', markersize=4)
plt.plot(iters, sample_count2, label='novel sample ED', marker='s', markersize=4)

# 添加标签和图例
plt.xlabel('iter')
plt.ylabel('ED')
plt.title('Energy Discrepancy over Iterations')
plt.legend()

# 显示图形
plt.grid(True)
plt.tight_layout()
plt.savefig('draw/ED_plot_clip_zs.png', dpi=300)
plt.show()
