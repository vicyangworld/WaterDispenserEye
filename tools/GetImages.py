# -*- coding: utf-8 -*-
import cv2  
import sys
import os

# 定义旋转rotate函数
def rotate(image, angle, center=None, scale=1.0):
    # 获取图像尺寸
    (h, w) = image.shape[:2]
    print(image.shape)

    # 若未指定旋转中心，则将图像中心设为旋转中心
    if center is None:
        center = (w / 2, h / 2)

    # 执行旋转
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, M, (w, h))

    # 返回旋转后的图像
    return rotated

def GetImagesFromVideo(windowName, picNum, savePath, cameraID=0, videoInput="", bRotate=True,value=-90):
	cv2.namedWindow(windowName)
	if videoInput=="":
		cap = cv2.VideoCapture(cameraID) 
	else:
		cap = cv2.VideoCapture(videoInput) #读入视频文件

	classfier = cv2.CascadeClassifier("/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt2.xml")
	recgColor = (0, 255, 0)

	count = 0  

	while cap.isOpened():
		ok,frame = cap.read()
		if not ok:
			break

		if bRotate:
			frame = rotate(frame,value)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faceRects = classfier.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
		if len(faceRects) > 0:
			for faceRect in faceRects:
				x,y,w,h = faceRect
				imgName = os.path.join(savePath,"A"+str(count)+".jpg")
				# imgName = savePath+"/"+str(count)+".jpg"
				image = frame[y-15:y+h+15,x-15:x+w+15] #截取人脸图像
				cv2.imwrite(imgName, image)#将图像写出到savePath

				count = count + 1
				if count > picNum:
					break
				cv2.rectangle(frame,(x-15,y-15),(x+w+15,y+h+15),recgColor, 3)
				font = cv2.FONT_HERSHEY_SIMPLEX
				cv2.putText(frame,str(count)+"/"+str(picNum),(x+15,y+15),font,1,(255,0,255),3)
		if count > picNum:
			break

		cv2.imshow(windowName, frame)
		kb = cv2.waitKey(1)
		if kb&0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	picNum = 2000
	savePath = '../data/'
	#video = "/home/yagnwk/Projects/linzhiling.mp4"
	#两种方式：一种是从摄像头获取人脸
	#一种是从视频获取人脸
	GetImagesFromVideo("Get Images",  picNum, savePath, cameraID=1, bRotate=False)
	#GetImagesFromVideo("Get Images",  picNum, savePath, videoInput=video,bRotate=False)
