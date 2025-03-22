import matplotlib.pyplot as plt
import numpy as np

# 数据准备
datasets = ["CodeContest", "HumanEval"]  # 数据集名称
methods = ["Direct", "CoT", "Self Planning", "MapCoder", "Optimized MapCoder"]  # 方法名称

# 每种方法在两个数据集上的正确率（%）
accuracy = {
    "Direct": [12.12, 85.37],
    "CoT": [8.48, 87.2],
    "Self Planning": [14.55, 85.98],
    "MapCoder": [20.61, 89.02],
    "Optimized MapCoder": [22.42, 92.07],
}

# 颜色映射（参考上传图片的配色）
colors = ['#6baed6', '#74c476', '#9e9ac8', '#fd8d3c', '#fdae61']

# 设置柱状图参数
bar_width = 0.2  # 柱子宽度
gap = 0.3  # 数据集之间的间隔
x = np.arange(len(datasets)) * (1 + gap)

# 创建画布
fig, ax = plt.subplots(figsize=(12, 8))  # 宽 12 英寸，高 8 英寸


# 绘制柱状图（添加边框效果）
for i, method in enumerate(methods):
    bars = ax.bar(x + i * bar_width - (len(methods) - 1) * bar_width / 2, 
                  accuracy[method], bar_width, color=colors[i], edgecolor='black', linewidth=1.2, label=method)

    # 添加数值标签
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, height + 1, 
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10, color='black')

# 设置X轴标签
ax.set_xticks(x)
ax.set_xticklabels(datasets)

# 轴标签和标题
ax.set_ylabel('Accuracy (%)')
ax.set_xlabel('Dataset')
ax.set_title('Method Performance on Different Datasets')

# 添加图例（调整位置，使其美观）
ax.legend(loc="upper left", bbox_to_anchor=(1, 1))

# 调整布局
plt.tight_layout()

# 显示图表
plt.show()
