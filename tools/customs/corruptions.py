import math
import cv2
import numpy as np
import random
from scipy.ndimage.interpolation import map_coordinates
from skimage.filters import gaussian
import skimage as sk
from scipy.ndimage import zoom as scizoom


class Corruptions:

    @staticmethod
    def __disk(radius, alias_blur=0.1, dtype=np.float32):
        if radius <= 8:
            L = np.arange(-8, 8 + 1)
            ksize = (3, 3)
        else:
            L = np.arange(-radius, radius + 1)
            ksize = (5, 5)
        X, Y = np.meshgrid(L, L)
        aliased_disk = np.array((X ** 2 + Y ** 2) <= radius ** 2, dtype=dtype)
        aliased_disk /= np.sum(aliased_disk)

        # supersample disk to antialias
        return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)

    @staticmethod
    def apply_gaussian_noise(img, severity):
        c = [.08, .12, 0.18, 0.26, 0.38][severity - 1]

        x = np.array(img) / 255.
        return (np.clip(x + np.random.normal(size=x.shape, scale=c), 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_shot_noise(img, severity):
        c = [60, 25, 12, 5, 3][severity - 1]
        x = np.array(img) / 255.
        return (np.clip(np.random.poisson(x * c) / float(c), 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_impulse_noise(img, severity):
        c = [.03, .06, .09, 0.17, 0.27][severity - 1]

        x = sk.util.random_noise(np.array(img) / 255., mode='s&p', amount=c)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    @classmethod
    def apply_defocus_blur(cls, img, severity):
        c = [(3, 0.1), (4, 0.5), (6, 0.5), (8, 0.5), (10, 0.5)][severity - 1]

        x = np.array(img) / 255.
        kernel = cls.__disk(radius=c[0], alias_blur=c[1])

        channels = []
        for d in range(3):
            channels.append(cv2.filter2D(x[:, :, d], -1, kernel))
        channels = np.array(channels).transpose((1, 2, 0))  # 3x224x224 -> 224x224x3

        return (np.clip(channels, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_frosted_glass_blur(img, severity):
        # ***
        rows, cols, _ = img.shape
        severity *= 2
        # 创建一个和原始图像大小相同的空白图像
        result = np.zeros((rows, cols, 3), dtype=np.uint8)

        # 随机移动像素值
        for i in range(rows - severity):
            for j in range(cols - severity):
                rand_i = int(np.random.normal(0, severity / 2))
                rand_j = int(np.random.normal(0, severity / 2))
                if i + rand_i >= rows - severity:
                    rand_i = rows - severity - i - 1
                if j + rand_j >= cols - severity:
                    rand_j = cols - severity - j - 1
                result[i, j] = img[i + rand_i, j + rand_j]

        return result

    @staticmethod
    def apply_motion_blur(img, severity):
        # ***
        size = severity * 5
        # 创建运动模糊的核
        kernel = np.zeros((size, size))
        kernel[int((size - 1) / 2), :] = np.ones(size)
        kernel = kernel / size

        # 应用卷积来模拟运动模糊
        blurred_img = cv2.filter2D(img, -1, kernel)

        return blurred_img

    @classmethod
    def apply_zoom_blur(cls, img, severity):
        c = [np.arange(1, 1.02, 0.01),
             np.arange(1, 1.04, 0.01),
             np.arange(1, 1.06, 0.02),
             np.arange(1, 1.08, 0.02),
             np.arange(1, 1.10, 0.02)][severity - 1]

        x = (np.array(img) / 255.).astype(np.float32)
        out = np.zeros_like(x)
        for zoom_factor in c:
            out += cls.__clipped_zoom(x, zoom_factor)

        x = (x + out) / (len(c) + 1)
        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def __clipped_zoom(img, zoom_factor):
        h = img.shape[0]
        w = img.shape[1]
        # ceil crop height(= crop width)
        ch = int(np.ceil(h / float(zoom_factor)))
        cw = int(np.ceil(w / float(zoom_factor)))

        top = (h - ch) // 2
        right = (w - cw) // 2
        img = scizoom(img[top:top + ch, right:right + cw], (zoom_factor, zoom_factor, 1), order=1)
        # trim off any extra pixels
        trim_top = (img.shape[0] - h) // 2
        trim_right = (img.shape[1] - w) // 2

        return img[trim_top:trim_top + h, trim_right:trim_right + w]

    @staticmethod
    def apply_snow(img, severity):
        snow_count = severity * random.randint(300, 500)
        img_with_snow = np.copy(img)
        rows, cols, _ = img.shape

        for _ in range(snow_count):
            x = random.randint(0, cols - 1)
            y = random.randint(0, rows - 1)
            size = random.randint(1, severity)  # 根据程度参数随机确定雪花的大小
            # 在随机位置添加不同大小的雪花点
            img_with_snow[max(0, y - size):min(rows, y + size), max(0, x - size):min(cols, x + size)] = [255, 255, 255]

        return img_with_snow

    @staticmethod
    def apply_rain(img, severity):
        length = 10 * severity
        angle = random.randint(-45, 60)
        w = 2 * severity - 1
        '''
        ***
        将噪声加上运动模糊,模仿雨滴
        https://blog.csdn.net/u014070279/article/details/108128452
        >>>输入
        length: 对角矩阵大小，表示雨滴的长度
        angle： 倾斜的角度，逆时针为正
        w:      雨滴大小
    
        >>>输出带模糊的噪声
    
        '''
        value = max(30, severity * 20)
        # 创建图像副本以避免在原始图像上操作
        rows, cols, _ = img.shape

        noise = np.random.uniform(0, 256, img.shape[0:2])
        # 控制噪声水平，取浮点数，只保留最大的一部分作为噪声
        v = value * 0.01
        noise[np.where(noise < (256 - v))] = 0

        # 噪声做初次模糊
        k = np.array([[0, 0.1, 0],
                      [0.1, 8, 0.1],
                      [0, 0.1, 0]])

        noise = cv2.filter2D(noise, -1, k)

        # 这里由于对角阵自带45度的倾斜，逆时针为正，所以加了-45度的误差，保证开始为正
        trans = cv2.getRotationMatrix2D((length / 2, length / 2), angle - 45, 1 - length / 100.0)
        dig = np.diag(np.ones(length))  # 生成对焦矩阵
        k = cv2.warpAffine(dig, trans, (length, length))  # 生成模糊核
        k = cv2.GaussianBlur(k, (w, w), 0)  # 高斯模糊这个旋转后的对角核，使得雨有宽度

        # k = k / length                         #是否归一化

        blurred = cv2.filter2D(noise, -1, k)  # 用刚刚得到的旋转后的核，进行滤波

        # 转换到0-255区间
        cv2.normalize(blurred, blurred, 0, 255, cv2.NORM_MINMAX)
        rain = np.array(blurred, dtype=np.uint8)

        # 输入雨滴噪声和图像
        beta = 0.2 + 0.05 * severity  # results weight
        # 显示下雨效果

        # expand dimensin
        # 将二维雨噪声扩张为三维单通道
        # 并与图像合成在一起形成带有alpha通道的4通道图像
        rain = np.expand_dims(rain, 2)
        rain_effect = np.concatenate((img, rain), axis=2)  # add alpha channel

        rain_result = img.copy()  # 拷贝一个掩膜
        rain = np.array(rain, dtype=np.float32)  # 数据类型变为浮点数，后面要叠加，防止数组越界要用32位
        rain_result[:, :, 0] = rain_result[:, :, 0] * (255 - rain[:, :, 0]) / 255.0 + beta * rain[:, :, 0]
        rain_result[:, :, 1] = rain_result[:, :, 1] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
        rain_result[:, :, 2] = rain_result[:, :, 2] * (255 - rain[:, :, 0]) / 255 + beta * rain[:, :, 0]
        # 对每个通道先保留雨滴噪声图对应的黑色（透明）部分，再叠加白色的雨滴噪声部分（有比例因子）
        return rain_result

    @staticmethod
    def apply_fog(img, severity):
        # ***
        img_f = img.copy() / 255.0
        (row, col, chs) = img.shape

        A = 0.8 - severity * 0.04  # 亮度
        beta = severity * 0.01  # 雾的浓度
        size = math.sqrt(max(row, col))  # 雾化尺寸
        # center = (random.randint(1,row-1), random.randint(1,col-1))  # 雾化中心
        # A = 0.5  # 亮度
        # beta = 0.08  # 雾的浓度
        # size = math.sqrt(max(row, col))  # 雾化尺寸
        center = (row // 2, col // 2)  # 雾化中心
        for j in range(row):
            for l in range(col):
                d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
                td = math.exp(-beta * d)
                img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
        return np.array(img_f * 255, dtype=np.uint8)

    @staticmethod
    def apply_brightness(img, severity):
        c = [.1, .2, .3, .4, .5][severity - 1]

        x = np.array(img) / 255.
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

        return (np.clip(x, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_contrast(img, severity):
        c = [.9, 1.2, 0.6, 1.5, .3][severity - 1]

        x = np.array(img) / 255.
        means = np.mean(x, axis=(0, 1), keepdims=True)
        return (np.clip((x - means) * c + means, 0, 1) * 255).astype(np.uint8)

    @staticmethod
    def apply_elastic(img, severity=1):
        c = [(244 * 0.02, 244 * 0.01, 244 * 0.02),
             (244 * 0.05, 244 * 0.01, 244 * 0.02),
             (244 * 0.08, 244 * 0.01, 244 * 0.02),
             (244 * 0.11, 244 * 0.01, 244 * 0.02),
             (244 * 0.14, 244 * 0.01, 244 * 0.02)][severity - 1]

        img = np.array(img, dtype=np.float32) / 255.
        shape = img.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32([center_square + square_size,
                           [center_square[0] + square_size, center_square[1] - square_size],
                           center_square - square_size])
        pts2 = pts1 + np.random.uniform(-c[2], c[2], size=pts1.shape).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        img = cv2.warpAffine(img, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)

        dx = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dy = (gaussian(np.random.uniform(-1, 1, size=shape[:2]),
                       c[1], mode='reflect', truncate=3) * c[0]).astype(np.float32)
        dx, dy = dx[..., np.newaxis], dy[..., np.newaxis]

        x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1)), np.reshape(z, (-1, 1))
        return np.clip(map_coordinates(img, indices, order=1, mode='reflect').reshape(shape), 0, 1) * 255

    @staticmethod
    def apply_pixelate(img, severity=1):
        # ***
        height, width = img.shape[:2]
        block_size = (int)(min(height, width) / severity / 1.25)

        # 将图像缩小到块大小的整数倍
        temp = cv2.resize(img, (block_size, block_size), interpolation=cv2.INTER_LINEAR)
        small = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        return small

    @staticmethod
    def apply_jpeg_compression(img, severity):
        # ***
        quality = [25, 18, 15, 10, 7][severity - 1]
        # 将图像编码为 JPEG 格式
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, buffer = cv2.imencode('.jpg', img, encode_param)

        # 解码 JPEG 图像
        reconstructed_img = cv2.imdecode(buffer, cv2.IMREAD_COLOR)

        return reconstructed_img

corruption_methods = {
    'guassian_noise': Corruptions.apply_gaussian_noise,
    'shot_noise': Corruptions.apply_shot_noise,
    'impulse_noise': Corruptions.apply_impulse_noise,
    'defocus_blur': Corruptions.apply_defocus_blur,
    'frosted_glass_blur': Corruptions.apply_frosted_glass_blur,
    'motion_blur': Corruptions.apply_motion_blur,
    'zoom_blur': Corruptions.apply_zoom_blur,
    'snow': Corruptions.apply_snow,
    'rain': Corruptions.apply_rain,
    'fog': Corruptions.apply_fog,
    'brightness': Corruptions.apply_brightness,
    'contrast': Corruptions.apply_contrast,
    'elastic': Corruptions.apply_elastic,
    'pixelate': Corruptions.apply_pixelate,
    'jpeg': Corruptions.apply_jpeg_compression
}

if __name__ == "__main__":
    # 加载图像
    img = cv2.imread('img.jpg')  # 替换成你的图像路径
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将图像从 BGR 色彩空间转换为 RGB

    # 定义每种处理方式的不同强度
    severitys = [1, 2, 3, 4, 5]  # 可根据需要调整强度范围

    # 应用各种不同处理方式及其强度
    processed_imgs = []
    for severity in severitys:
        noisy_img_impulse = Corruptions.apply_motion_blur(img, severity)
        processed_imgs.append(noisy_img_impulse)

    # 显示或保存处理后的图像
    for i, processed_img in enumerate(processed_imgs):
        cv2.imwrite(f'processed_img_{i + 1}.jpg', cv2.cvtColor(processed_img, cv2.COLOR_RGB2BGR))  # 保存处理后的图像