# -*- coding:utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from keras.models import load_model
from keras.utils import plot_model
from Dataset import Dataset
from Dataset import IMAGE_SIZE
import cv2


MODEL_PATH='./yangwk.face.model.h5'
class Model(object):
	"""docstring for Model"""
	def __init__(self):
		self.model = None

	def build(self,dataset,nb_classes=3):
		self.model = Sequential()
		#31，filters：卷积核的数目（即输出的维度），filter的一般为奇数
		self.model.add(Conv2D(31,(3,3),padding='same',input_shape=dataset.input_shape,activation='relu'))

		self.model.add(Conv2D(31,(3,3),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))

		self.model.add(Conv2D(63,(3,3),padding='same',activation='relu'))
		self.model.add(Conv2D(63,(3,3),activation='relu'))
		self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Dropout(0.25))


		self.model.add(Flatten())
		self.model.add(Dense(512, activation='relu'))
		#self.model.add(Dropout(0.5))
		self.model.add(Dense(nb_classes,activation='softmax'))

		self.model.summary()

	# data_augmentation:数据扩充,
	def train(self, dataset, batch_size=40, nb_epoch=10, data_augmentation=True):
		# 优化器：随机梯度下降，对损失函数进行求解（训练模型，即调整训练参数：权重和偏置值）使其最优，确保e值最小
		# decay指定每次更新后学习效率的衰减值，一般会选择0.005
		# momentum指定动量值(一般选择0.9-0.95之间)，模拟物体运动时的惯性，让优化器在一定程度上保留之前的优化方向，  ...
		# 同时利用当前样本微调最终的优化方向，这样即能增加稳定性，提高学习速度，又在一定程度上避免了陷入局部最优陷阱;
		# nesterov则用于指定是否采用nesterov动量方法，nesterov momentum是对传统动量法的一个改进方法，其效率更高。
		sgd = SGD(lr=0.01, decay=0.0005, momentum=0.9, nesterov=True)
		# comile()完成训练配置，之后便开始训练fit; loss：损失函数，“categorical_crossentropy”，常用于多分类问题
		# 参数metrics用于指定模型评价指标，参数值”accuracy“表示用准确率来评价
		# 代码中loss的值为“categorical_crossentropy”，常用于多分类问题，其与激活函数softmax配对使用...
		# 类别只有两种，也可采用‘binary_crossentropy’二值分类函数，该函数与sigmoid配对使用，注意如果采用它dataset中就不需要one-hot编码
		self.model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

		# data_augmentation:数据扩充,指的是使用一些方法（旋转，翻转，移动等）来增加数据输入量。这里，特指图像数据。
		# 通过Data Aumentation防止过拟合的问题
		if not data_augmentation:
			# batch_size：整数，指定进行梯度下降时每个batch包含的样本数。训练时一个batch的样本会被计算一次梯度下降，使目标函数优化一步。...
			# batch_size则是一个影响模型训练结果的重要参数
			# epochs 训练的总轮数,1个epoch等于使用训练集中的全部样本训练一次
			# 还有一个iteration的概念，指的是一个epoch里需要多少次batch_size才能将训练集轮一次
			# shuffle：布尔值或字符串，一般为布尔值，表示是否在训练过程中随机打乱输入样本的顺序。...
			# 若为字符串“batch”，则是用来处理HDF5数据的特殊情况，它将在batch内部将数据打乱。
			self.model.fit( dataset.train_images,dataset.train_labels,
							batch_size = batch_size, epochs=nb_epoch, 
							validation_data=(dataset.valid_images,dataset.valid_labels),
							shuffle=True )
		else:
			# 使用ImageDataGenerator类实现数据扩充，keras的自带库
			# 这里相当于新建了一个配置文件（ImageDataGenerator对象dategen）,说明要做哪些具体工作来实现数据扩充
			datagen = ImageDataGenerator(
						featurewise_center = False,             #是否使输入数据去中心化（均值为0），
						samplewise_center  = False,             #是否使输入数据的每个样本均值为0
						featurewise_std_normalization = False,  #是否数据标准化（输入数据除以数据集的标准差）
						samplewise_std_normalization  = False,  #是否将每个样本数据除以自身的标准差
						zca_whitening = False,                  #是否对输入数据施以ZCA白化
						rotation_range = 20,                    #数据提升时图片随机转动的角度(范围为0～180)
						width_shift_range  = 0.2,               #数据提升时图片水平偏移的幅度（单位为图片宽度的占比，0~1之间的浮点数）
						height_shift_range = 0.2,               #同上，只不过这里是垂直
						horizontal_flip = True,                 #是否进行随机水平翻转
						vertical_flip = True )                  #是否进行随机垂直翻转

			# 将待扩充数据集按照datagen（配置文件）都配置好
			datagen.fit(dataset.train_images)                        
			# flow()将会返回一个生成器，这个生成器用来扩充数据，每次都会产生batch_size个样本。 
			#  fit_generator使得训练时不用将数据全部加入内存，生成器与模型将并行执行以提高效率。...
			#  例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
			# 利用生成器开始训练模型
			# steps_per_epoch的含义是一个epoch分成几个batch_size, 也就是每轮的iteration, allTrainsample/batch_size, ...
			# 比如，有1000个训练样本，steps_per_epoch = 10的话，那么就是一轮迭代十次，batch_size=100
			self.model.fit_generator(datagen.flow(dataset.train_images, dataset.train_labels,batch_size = batch_size),
									 #steps_per_epoch = dataset.train_images.shape[0],
									 samples_per_epoch = dataset.train_images.shape[0],
									 epochs = nb_epoch,
									 validation_data = (dataset.valid_images, dataset.valid_labels))

	def evaluate(self,dataset):
		# verbose：日志显示，0为不在标准输出流输出日志信息，1为输出进度条记录，2为每个epoch输出一行记录
		score = self.model.evaluate(dataset.test_images,dataset.test_labels, verbose=1)
		print("%s: %.2f%%" % (self.model.metrics_names[1], score[1]*100))

	def save(self, file_path=MODEL_PATH):
		self.model.save(file_path)

	def load(self,file_path=MODEL_PATH):
		self.model = load_model(file_path)

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

	def face_predict(self,image):
		if K.image_dim_ordering() == 'th' and image.shape!=(1,3,IMAGE_SIZE,IMAGE_SIZE):
			image = self.__resize_image(image)
			image = image.reshape((1,3,IMAGE_SIZE,IMAGE_SIZE))
		elif K.image_dim_ordering() == 'tf' and image.shape!=(1,IMAGE_SIZE,IMAGE_SIZE,3):
			image = self.__resize_image(image)
			image = image.reshape(1,IMAGE_SIZE,IMAGE_SIZE,3)
		else:
			pass

		image = image.astype('float32')
		image /= 255
		#result包含每个类别的概率
		proba = self.model.predict_proba(image)
		print('proba: ', proba)
		#c = input()
		#result是一个one-hot向量，向量的长度为类别的数量
		result = self.model.predict_classes(image)
		print('result: ', result)
		#分类是自己 的位于在one-hot向量中的第一个元素，如果为0,则为自己，label就是这么设置的
		return result[0]

if __name__ == '__main__':
	dataset = Dataset('./data/')
	dataset.load(nb_classes=3)

	model = Model()
	# for train
	model.build(dataset,nb_classes=3)
	model.train(dataset)
	model.save(file_path='./model/face.model.h5')
	# for test	
	model.load(file_path='./model/face.model.h5')
	model.evaluate(dataset)