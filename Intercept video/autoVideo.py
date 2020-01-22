import cv2
import os

outputPath='E:/pyproject/MCCV/Intercept video/'

vc=cv2.VideoCapture('E:/pyproject/MCCV/Intercept video/bandicam 2019-03-25 22-49-40-918.mp4')
c=1

if vc.isOpened():
    rval,frame=vc.read()
else:
    rval=False

timeF=15

while rval:
    rval,frame=vc.read()
    if(c%timeF==0):
        cv2.imwrite(outputPath+'image'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()

def mywalk(path):
    # path为绝对路径
    # 创建一个空列表，用于存储文件的绝对路径
    files_list = []
    # 遍历给定路径下所有的元素（这些元素可能是目录，也可能是文件）
    for element in os.listdir(path):
            files_list += [os.path.join(path, element)]
    # 返回最终的列表
    return files_list

a=mywalk(outputPath)

for i in a:
    img = cv.imread(i)
    dim=(512,512)
    # 变成指定尺寸
    new_image = cv.resize(img,dim)
    # 进行存储处理后的图片
    cv.imwrite(i,new_image)
    # 比例对的图片覆盖以前比例不对的图片