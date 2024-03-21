import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm

# 指定字体
font_path = 'simsun.ttc'  # 这里也可以填其他字体的绝对路径，只填名称表示本程序与该字体在同一级目录下
legend_font_prop = fm.FontProperties(fname=font_path, size=10)
title_font_prop = fm.FontProperties(fname=font_path, size=14)

colors = ['#4472C4', '#CC8965', '#7C7C7C', '#FFD966', '#857BAB', '#609E6F']
cors = ['mRCE', '噪声', '模糊', '天气', '数字失真']


def plot(data, models):
    # 绘制柱状图
    fig, ax = plt.subplots()
    width = 0.14  # 柱状图的宽度
    spacing = 0.02  # 柱状图之间的间距
    
    # 遍历每个属性，绘制柱状图
    for i, attr in enumerate(cors):
        x = np.arange(len(models)) + (width + spacing) * i
        ax.bar(x, data[:, i], width=width, label=attr, color=colors[i], )

    # 设置横轴标签和标题
    ax.set_xticks(np.arange(len(models)) + width * (len(cors) - 1) / 2)
    ax.set_xticklabels(models, rotation=45, fontproperties=legend_font_prop)
    ax.set_ylabel('相对损坏误差（RCE）', fontproperties=title_font_prop)
    ax.set_ylim(0, 60)
    ax.set_yticks(np.arange(0, 61, 10))
    ax.set_yticklabels([f'{i}%' for i in np.arange(0, 61, 10)])
    
    # 隐藏上方和右方的坐标轴线
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='-.', color='gray')
    ax.set_axisbelow(True)
    # 添加图例
    ax.legend(labels=cors, loc='upper left', bbox_to_anchor=(0, 1), ncol=3, prop=legend_font_prop)
    ax.axhline(y=data[-1, 0], color='black', linestyle='--', linewidth=1)

    # 调整图形布局，防止图例被遮挡
    plt.subplots_adjust(bottom=0.2, left=0.15)
    plt.savefig(f"模型相对损坏误差.png")
    # 显示图形
    plt.show()


if __name__ == "__main__":
    # 神经网络模型和属性名称
    models = ['YOLOv5', 'YOLOv6', 'YOLOv7', 'YOLOv8', 'YOLOv8-FS', 'YOLOv8-FA']
    # 随机生成数据，每个模型有四个属性的值
    data = np.array([[17.65, 26.31, 12.5,  23.34, 10.6],
                     [20.96, 42.35, 17.65, 19.39, 9.8],
                     [14.48, 30.24, 16.93, 8.86,  5.84],
                     [17.27, 35.57, 11.44, 17.21, 9.44],
                     [16.39, 31.94, 11.52, 16.9, 9.09],
                     [14.65, 28.24, 9.71,  15.4, 8.65]])
    plot(data=data, models=models)
