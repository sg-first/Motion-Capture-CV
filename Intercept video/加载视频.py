#!usr/bin/python

import cv2

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
        cv2.imwrite('E:/pyproject/MCCV/Intercept video/image'+str(c)+'.jpg',frame)
    c=c+1
    cv2.waitKey(1)
vc.release()