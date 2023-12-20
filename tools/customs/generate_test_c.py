from corruptions import corruption_methods
import cv2
import os
from PIL import Image
import numpy as np
from imagecorruptions import corrupt


corruptions_group = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                     'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                     'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                     'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']


def generate_c_standard(ori_path, out_path):
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    for i in range(len(corruptions_group)):
        dir = os.path.join(out_path, corruptions_group[i])
        if not os.path.exists(dir):
            os.mkdir(dir)
        for severity in range(5):
            sub_dir = os.path.join(dir, str(severity + 1))
            if not os.path.exists(sub_dir):
                os.mkdir(sub_dir)
            # 加载图像
            for item in os.listdir(ori_path):
                dst_img_path = os.path.join(sub_dir, item)
                if item.startswith('.') or not item.endswith('.jpg') or os.path.exists(dst_img_path):
                    continue
                img_path = os.path.join(ori_path, item)
                img = np.asarray(Image.open(img_path))
                corrupted_img = corrupt(img, corruption_number=i, severity=severity + 1)
                cv2.imwrite(dst_img_path, corrupted_img)  # 保存处理后的图像


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