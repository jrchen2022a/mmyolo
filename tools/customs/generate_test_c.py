from corruptions import corruption_methods
import cv2
import os


def generate_c(ori_path, out_path, corruptions=corruption_methods.keys(), severities=(1, 2, 3, 4, 5)):
    # 加载图像
    for item in os.listdir(ori_path):
        if item.startswith('.') or not item.endswith('.jpg'):
            continue
        img_path = os.path.join(ori_path, item)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 色彩空间转换为 RGB
        for ct in corruptions:
            dir = os.path.join(out_path,ct)
            if not os.path.exists(dir):
                os.mkdir(dir)
            for severity in severities:
                sub_dir = os.path.join(dir,str(severity))
                if not os.path.exists(sub_dir):
                    os.mkdir(sub_dir)
                corrupt_img = corruption_methods[ct](img, severity)
                dst_img_path = os.path.join(sub_dir, item)
                cv2.imwrite(dst_img_path, cv2.cvtColor(corrupt_img, cv2.COLOR_RGB2BGR))  # 保存处理后的图像


if __name__ == "__main__":
    ori_path = r'/Volumes/T7/researches/code/mmyolo/tools/customs'
    out_path = r'/Volumes/T7/researches/code/mmyolo/tools/customs'
    generate_c(ori_path, out_path)