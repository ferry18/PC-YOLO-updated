import cv2
import numpy as np

#  摸模拟雨滴在摄像头上

# 读取图像
image_path = "D:\way\yolo\Datasets\datasets\images\\train\\bremen\\bremen_000310_000019_leftImg8bit.png"   # 替换为你的图像路径
image = cv2.imread(image_path)

# 转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用高斯模糊减少噪声
blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

# 计算图像的拉普拉斯算子，以突出边缘
laplacian_image = cv2.Laplacian(blurred_image, cv2.CV_64F)

# 将拉普拉斯图像扩展到与原图相同的颜色通道数
laplacian_image_3channel = cv2.merge([laplacian_image, laplacian_image, laplacian_image])

# 将拉普拉斯图像与原图相加，增加清晰度
enhanced_image = cv2.convertScaleAbs(image - 0.5 * laplacian_image_3channel)

# 增加对比度
enhanced_image = cv2.convertScaleAbs(enhanced_image * 1.5)

# 保存增强后的图像
cv2.imwrite('D:\way\yolo\enhced_image.jpg', enhanced_image)  # 保存图像