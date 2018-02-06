# -*- coding:utf-8 -*-
import random
from sklearn.cross_validation import train_test_split
from keras import backend as K
from keras.utils import np_utils
import numpy as np
import os
import cv2

IMAGE_SIZE = 64
class Dataset:
	def __init__(self, path_name):
		#train
		self.train_images = None
		self.train_labels = None
		#valid
		self.valid_images = None
		self.valid_labels = None
		#test
		self.test_images  = None
		self.test_labels  = None

		self.path_name = path_name
		#当前库采用的维度顺序
		self.input_shape = None
		
	#加载数据集并按照交叉验证的原则划分数据集并进行相关预处理工作
	def load(self, img_rows = IMAGE_SIZE, img_cols = IMAGE_SIZE, img_channels = 3, nb_classes = 3):
		self.__images = []
		self.__labels = []
		#加载数据集到内存
		#将输入的所有图片转成四维数组，尺寸为(图片数量*IMAGE_SIZE*IMAGE_SIZE*通道数3)
		self.__load_dataset()
		#train_test_split 是sklearn中用来划分训练集与测试集的，random_state：是随机数的种子。
		#随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。
		train_images,valid_images,train_labels,valid_labels = train_test_split(self.__images,self.__labels,test_size=0.3, random_state=random.randint(0,100))
		_, test_images,_,test_labels = train_test_split(self.__images,self.__labels,test_size=0.5,random_state=random.randint(0,100))
		
		#判断keras的后端，这主要涉及到图片的存储结构，theno是：颜色通道数 × 行数 × 列数
		#tensorflow 是：行 × 列 × 通道数
		if K.image_dim_ordering()=='th':
			train_images = train_images.reshape(train_images.shape[0],img_channels, img_rows, img_cols)
			valid_images = valid_images.reshape(valid_images.shape[0],img_channels, img_rows, img_cols)
			test_images  = test_images.reshape(test_images.shape[0].img_channels,img_rows,img_cols)
			self.input_shape = (img_channels,img_rows,img_cols)
		else:
			train_images = train_images.reshape(train_images.shape[0],img_rows,img_cols,img_channels)
			valid_images = valid_images.reshape(valid_images.shape[0],img_rows,img_cols,img_channels)
			test_images = test_images.reshape(test_images.shape[0],img_rows,img_cols,img_channels)
			self.input_shape = (img_rows,img_cols,img_channels)
		
		print(train_images.shape[0],'train samples')       
		print(valid_images.shape[0],'valid samples')
		print(test_images.shape[0],'test samples')
		
		#将输出向量化，比如输出是两类，那么label = (class1, class2)
		#compile()的categorical_crossentropy函数要求标签集必须采用one-hot编码形式

		train_labels = np_utils.to_categorical(train_labels, nb_classes)
		valid_labels = np_utils.to_categorical(valid_labels, nb_classes)
		test_labels = np_utils.to_categorical(test_labels, nb_classes)

		#将图像的像素值转换成浮点数
		train_images = train_images.astype('float32')
		valid_images = valid_images.astype('float32')
		test_images = test_images.astype('float32')
		#归一化
		train_images /= 255
		valid_images /= 255
		test_images /= 255

		self.train_images = train_images
		self.valid_images = valid_images
		self.test_images  = test_images
		self.train_labels = train_labels
		self.valid_labels = valid_labels
		self.test_labels  = test_labels


	def __resize_image(self, image, height=IMAGE_SIZE, width=IMAGE_SIZE):
		top,bottom,left,right = (0, 0, 0, 0)
		h, w, _ = image.shape
		maxx = max(h,w)
		if h < maxx:
			dh = maxx - h
			top = dh // 2
			bottom = dh - top
		if w < maxx:
			dw = maxx - w
			left = dw // 2
			right = dw - left
		#cv2中的补边方法，cv2.BORDER_CONSTANT意思是以常量value填充,这里的value表示黑色
		constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0,0,0])
		#重新设置图片大小
		return cv2.resize(constant, (height,width))

	def __read_images(self,path_name):
		for dir_item in os.listdir(path_name):
			full_path = os.path.abspath(os.path.join(path_name, dir_item))
			if os.path.isdir(full_path):
				self.__read_images(full_path)
			else:
				if dir_item.endswith('.jpg'):
					image = cv2.imread(full_path)
					image = self.__resize_image(image)
					self.__images.append(image)
					self.__labels.append(path_name)

	def __load_dataset(self):
		self.__read_images(self.path_name)
		self.__images = np.array(self.__images)
		print("Size of a data: "+str(self.__images.shape))
		# yangwk:0  liuzl:1 other:2
		#----- 
		label_list=[]
		for label in self.__labels:
			if label.endswith('yangwk'):
				label_list.append(0)
			if label.endswith('linzl'):
				label_list.append(1)
			if label.endswith('other'):
				label_list.append(2)
		self.__labels = np.array(label_list)
		#-----
		# self.__labels = np.array([0 if label.endswith('yangwk') else 1 for label in self.__labels])
		#print(self.__labels)


if __name__ == '__main__':
	data = Dataset('./data/')
	data.load(nb_classes=3)