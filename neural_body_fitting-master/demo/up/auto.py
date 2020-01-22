import cv2 as cv
import os

def mywalk(path):
    # path为绝对路径
    # 创建一个空列表，用于存储文件的绝对路径
    files_list = []
    # 遍历给定路径下所有的元素（这些元素可能是目录，也可能是文件）
    for element in os.listdir(path):
            files_list += [os.path.join(path, element)]
    # 返回最终的列表
    return files_list

a=mywalk("E:/pyproject/MCCV/neural_body_fitting-master/demo/up/input")

for i in a:
    img = cv.imread(i)
    dim=(512,512)
# 变成指定尺寸
    new_image = cv.resize(img,dim)
# 进行存储处理后的图片
    cv.imwrite(i,new_image)
#比例对的图片覆盖以前比例不对的图片





