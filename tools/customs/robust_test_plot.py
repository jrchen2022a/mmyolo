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
    plt.plot(x, y_clean_pre, marker='o', color='b', label='干净测试集')
    plt.plot(x, y_noise_pre, marker='o', color='g', label='噪声测试集')

    # 计算x和y的均值
    mean_clean_pre = np.mean(y_clean_pre)
    std_clean_pre = np.std(y_clean_pre)
    mean_noise_pre = np.mean(y_noise_pre)
    std_noise_pre = np.std(y_noise_pre)

    # 打印均值
    print(f"{model_name} 干净数据集  均值：{mean_clean_pre:.2f}  标准差：{std_clean_pre:.2f}")
    print(f"{model_name} 噪声数据集  均值：{mean_noise_pre:.2f}  标准差：{std_noise_pre:.2f}")
    plt.axhline(y=mean_clean_pre, color='b', linestyle='--')
    plt.axhline(y=mean_noise_pre, color='g', linestyle='--')

    # 在图上添加均值标注
    plt.annotate(f'均值：{mean_clean_pre:.2f}%', xy=(4.5, mean_clean_pre), xytext=(-10, 15),
                 fontproperties=legend_font_prop, textcoords='offset points', arrowprops=dict(arrowstyle='-'))
    plt.annotate(f'均值：{mean_noise_pre:.2f}%', xy=(4.5, mean_noise_pre), xytext=(-10, 15),
                 fontproperties=legend_font_prop, textcoords='offset points', arrowprops=dict(arrowstyle='-'))

    # 添加标题、x轴和y轴标签
    title = f"{model_name}网络的抗噪性能表现"
    plt.title(title, fontproperties=title_font_prop)
    plt.xlabel("拟合网络", fontproperties=title_font_prop)
    plt.ylabel("mAP/%", fontproperties=title_font_prop)
    plt.xticks(ticks=x, labels=x)
    plt.ylim(50, 100)
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
                        help='干净测试集表现')
    parser.add_argument('--noise', type=float, nargs='+',
                        help='噪声测试集表现')

    # 解析命令行参数
    args = parser.parse_args()
    assert len(args.clean) == 5 and len(args.noise) == 5
    plot([x*100 for x in args.clean], [x*100 for x in args.noise], args.model)


if __name__ == '__main__':
    main()