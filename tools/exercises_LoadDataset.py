# -*- coding: utf-8 -*-

import os
import sys
import numpy as np


IMAGE_SIZE = 64

#按照指定图像大小调整尺寸
def resize_image(image, height = IMAGE_SIZE, width = IMAGE_SIZE):
	top, bottom, left, right = (0, 0, 0, 0)
	#获取图像尺寸
	h, w, _ = image.shape
	#对于长宽不相等的图片，找到最长的一边
	longest_edge = max(h, w)
	#计算短边需要增加多上像素宽度使其与长边等长
	if h < longest_edge:
		dh = longest_edge - h
		top = dh // 2
		bottom = dh - top
	elif w < longest_edge:
		dw = longest_edge - w
		left = dw // 2
		right = dw - left
	else:
		pass
	BLACK = [0, 0, 0]
	#给图像增加边界，是图片长、宽等长，cv2.BORDER_CONSTANT指定边界颜色由value指定
	constant = cv2.copyMakeBorder(image, top , bottom, left, right, cv2.BORDER_CONSTANT, value = BLACK)
	#调整图像大小并返回
	return cv2.resize(constant, (height, width))

#读取训练数据
images = []
labels = []
def read_images(path_name):
	for dir_item in os.listdir(path_name):
		full_path = os.path.abspath(os.path.join(path_name, dir_item))
		if os.path.isdir(full_path):
			read_images(full_path)
		else:
			if dir_item.endswith('.jpg'):
				print(full_path)
				image = cv2.imread(full_path)
				image = resize_image(image, IMAGE_SIZE, IMAGE_SIZE)
				images.append(image)
				labels.append(path_name)
	return images,labels
	
#从指定路径读取训练数据
def load_dataset(path_name):
	images,labels = read_images(path_name)    
	#将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*3)
	#图片为64 * 64像素,一个像素3个颜色值(RGB)
	images = np.array(images)
	labels = np.array([0 if label.endswith('yangwk') else 1 for label in labels])
	return images, labels

if __name__ == '__main__':
	path_name = './data/'
	images, labels = load_dataset(path_name)
	print(images.shape)
	print(labels.shape)