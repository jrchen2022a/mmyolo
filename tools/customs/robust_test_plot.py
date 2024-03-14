import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import argparse

# 指定字体
font_path = 'simsun.ttc'  # 这里也可以填其他字体的绝对路径，只填名称表示本程序与该字体在同一级目录下
legend_font_prop = fm.FontProperties(fname=font_path, size=10)
title_font_prop = fm.FontProperties(fname=font_path, size=14)


def plot(y_clean_pre, y_noise_pre, model_name):
    x = [1, 2, 3, 4, 5]
    # 绘制散点图
    plt.scatter(x, y_clean_pre)
    plt.scatter(x, y_noise_pre)
    # 绘制折线图
    plt.plot(x, y_clean_pre, marker='o', color='b', label='原测试集')
    plt.plot(x, y_noise_pre, marker='o', color='g', label='噪声测试集')

    # 计算x和y的均值
    mean_clean_pre = np.mean(y_clean_pre)
    mean_noise_pre = np.mean(y_noise_pre)

    # 打印均值
    # plt.axhline(y=mean_clean_pre, color='b', linestyle='--', label='原测试集均值')
    plt.axhline(y=mean_noise_pre, color='g', linestyle='--')

    # 在图上添加均值标注
    plt.annotate(f'均值：{mean_noise_pre:.2f}', xy=(4.5, mean_noise_pre), xytext=(5, 15),
                 fontproperties=legend_font_prop, textcoords='offset points', arrowprops=dict(arrowstyle='-'))

    # 添加标题、x轴和y轴标签
    title = f"{model_name}对抗高斯噪声性能表现"
    plt.title(title, fontproperties=title_font_prop)
    plt.xlabel("拟合模型", fontproperties=title_font_prop)
    plt.ylabel("mAP", fontproperties=title_font_prop)
    plt.xticks(ticks=x, labels=x)
    plt.ylim(0.5, 1)
    plt.legend(prop=legend_font_prop)

    # 显示网格线
    plt.grid(True)
    # 显示图形
    plt.savefig(f"{model_name}.png")
    plt.show()


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='绘制模型的鲁棒实验测试图')

    parser.add_argument('--model', type=str, help='模型名称')
    parser.add_argument('--clean', type=float, nargs='+',
                        help='原测试集表现')
    parser.add_argument('--noise', type=float, nargs='+',
                        help='噪声测试集表现')

    # 解析命令行参数
    args = parser.parse_args()
    assert len(args.clean) == 5 and len(args.noise) == 5
    plot(args.clean, args.noise, args.model)


if __name__ == '__main__':
    main()

    # --model YOLOv8-FA --clean 0.869 0.87 0.868 0.871 0.87   --noise 0.713 0.708 0.713 0.704 0.696
    # --model YOLOv8-FS --clean 0.864 0.864 0.863 0.866 0.867 --noise 0.66 0.652 0.654 0.641 0.639
    # --model YOLOv8    --clean 0.861 0.864 0.863 0.862 0.864 --noise 0.623 0.689 0.608 0.672 0.657
