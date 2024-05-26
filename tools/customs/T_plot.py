import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import argparse


def plot(x, y):
    # 绘制散点图
    plt.scatter(x, y)
    # 绘制折线图
    plt.plot(x, y, marker='x', color='b')

    # 添加标题、x轴和y轴标签
    plt.xlabel("T")
    plt.ylabel("mAP (%)")
    plt.xticks(ticks=[1.0,1.5,2.0,2.5,3.0,3.5,4.0,4.5])
    plt.xticks(ticks=[82.0, 82.5, 83.0, 83.5, 84.0, 84.5])

    # 显示图形
    plt.savefig(f"T.png", dpi=300)
    plt.show()


def main():
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(description='绘制温度系数影响图')

    parser.add_argument('--x', type=float, nargs='+',
                        help='温度系数')
    parser.add_argument('--y', type=float, nargs='+',
                        help='模型表现')

    # 解析命令行参数
    args = parser.parse_args()
    plot(args.x, args.y)


if __name__ == '__main__':
    x_data = [1.0, 2.5, 3.7, 4.2, 5.6]
    y_data = [3.0, 4.5, 2.8, 5.1, 6.0]
    plt.scatter(x_data, y_data)
    plt.xlabel('X Axis')
    plt.ylabel('Y Axis')
    plt.title('Scatter Plot')
    plt.show()