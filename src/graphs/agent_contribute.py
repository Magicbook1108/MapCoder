import matplotlib.pyplot as plt
import pandas as pd

# 定义表格数据
cc_data = [
    ["✘", "✘", "✔", "18.18%", "2.43"],
    ["✔", "✘", "✔", "%", ""],
    ["✘", "✔", "✔", "%", ""],
    ["✔", "✔", "✘", "%", ""],
    ["✔", "✘", "✘", "%", ""],
    ["✘", "✔", "✘", "17.58%", "3.03%"],
    ["✔", "✔", "✔", "20.61%", "-"]
]

human_data = [
    ["✘", "✘", "✔", "87.2%", ""],
    ["✔", "✘", "✔", "86.59%", ""],
    ["✘", "✔", "✔", "89.63%", ""],
    ["✔", "✔", "✘", "87.2%", ""],
    ["✔", "✘", "✘", "87.8%", ""],
    ["✘", "✔", "✘", "89.63%", ""],
    ["✔", "✔", "✔", "89.02%", "-"]
]

data = cc_data

columns = ["Retrieval Agent", "Planning Agent", "Debugging Agent", "Pass@1", "Performance Drop"]

# 创建 Matplotlib 画布
fig, ax = plt.subplots(figsize=(8, 6))

# 隐藏坐标轴
ax.set_frame_on(False)
ax.xaxis.set_visible(False)
ax.yaxis.set_visible(False)

# 添加表格
table = ax.table(cellText=data, colLabels=columns, loc='center', cellLoc='center', colWidths=[0.2] * 5)

# 调整表格样式
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.2)  # 调整表格大小

# 显示表格
plt.show()
