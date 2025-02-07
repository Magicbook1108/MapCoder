import matplotlib.pyplot as plt
import numpy as np

# 数据准备
methods = ["Direct", "MapCoder",]  # 方法名称
acc =     [12.12,    20.61,  ]                        # 准确率（%）
total_tokens = [237818, 7170454, ]       # 总令牌数（转换为百万单位）
tokens_m = [x / 1e6 for x in total_tokens]      # 单位转换为百万（1.365M, 2.45M等）

# 创建画布和主Y轴
fig, ax1 = plt.subplots(figsize=(10, 6))

# 绘制准确率条形图（主Y轴）
bar_width = 0.35
x = np.arange(len(methods))  # X轴位置
bars_acc = ax1.bar(x - bar_width/2, acc, bar_width, color='#1f77b4', label='Accuracy (%)')

# 设置主Y轴标签和范围
ax1.set_ylabel('Accuracy (%)', color='#1f77b4')
ax1.set_ylim(10, 30)
ax1.tick_params(axis='y', labelcolor='#1f77b4')

# 创建次Y轴（共享同一X轴）
ax2 = ax1.twinx()

# 绘制总令牌数条形图（次Y轴）
bars_tokens = ax2.bar(x + bar_width/2, tokens_m, bar_width, color='#ff7f0e', label='Total Tokens (M)')

# 设置次Y轴标签和范围
ax2.set_ylabel('Total Tokens (M)', color='#ff7f0e')
ax2.set_ylim(0, max(tokens_m) * 1.5)  # 自动调整范围
ax2.tick_params(axis='y', labelcolor='#ff7f0e')

# 添加标题、X轴标签和图例
plt.title('performance on codecontests')
ax1.set_xticks(x)
ax1.set_xticklabels(methods)
fig.legend(loc="upper right", bbox_to_anchor=(0.9, 0.9))

# 显示数值标签
for bar in bars_acc:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, height, f'{height}%', ha='center', va='bottom')

for bar in bars_tokens:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height, f'{height:.2f}M', ha='center', va='bottom')

plt.tight_layout()
plt.show()
