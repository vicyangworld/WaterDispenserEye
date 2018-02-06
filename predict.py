# -*- coding: utf-8 -*-

import cv2
import sys
from Model import Model

def RecognizeFace(windownName,cameraID):
	model = Model()
	model.load(file_path='./model/face.model.h5')
	cv2.namedWindow(windownName)
	cap = cv2.VideoCapture(cameraID)
	#创建一个分类器对象，详见下面关于CascadeClassifier的原理简述...
	#haarcascade_frontalface_alt_tree.xml最为严格的人脸分类
	classfier = cv2.CascadeClassifier("/usr/local/lib/python3.6/dist-packages/cv2/data/haarcascade_frontalface_alt2.xml")
	#定义一个颜色变量
	# recColor = (0, 255, 0)
	while cap.isOpened():
		ok,frame = cap.read()
		if not ok:
			break
		#色彩转换，将读取到的一帧图画变成灰度图像，目的是降低计算量，提高检测速度
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#分类器对象的detectMultiScale方法可以检测出图片中所有的人脸，并将人脸用vector保存各个人脸的坐标、大小，...
		#返回的是一个vector
		faceRects = classfier.detectMultiScale(gray, scaleFactor = 1.2, minNeighbors = 3, minSize = (32, 32))
		if len(faceRects)>0:
			for faceRect in faceRects:
				x,y,w,h=faceRect
				image = frame[y-10:y+h+10,x-10:x+w+10] #截取人脸图像
				faceID = model.face_predict(image) #将人脸图像交给model
				if faceID==0:
					#画矩形五个参数，图像、两个对角点坐标、颜色数组、线宽
					cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,255,0),2)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(frame,"yangwk",(x+30,y+30),font,1,(255,0,255),3)
				if faceID==1:
					cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(255,0,0),2)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(frame,"linzl",(x+30,y+30),font,1,(255,0,255),3)			
				if faceID==2:
					cv2.rectangle(frame,(x-10,y-10),(x+w+10,y+h+10),(0,0,255),2)
					font = cv2.FONT_HERSHEY_SIMPLEX
					cv2.putText(frame,"other",(x+30,y+30),font,1,(255,0,255),3)				

		cv2.imshow(windownName,frame)
		keyboard = cv2.waitKey(10)
		if keyboard&0xFF == ord('q'):
			break

	cap.release()
	cv2.destroyAllWindows()

if __name__ == '__main__':
	cameraID = 1
	RecognizeFace("Recognizing faces",cameraID)
