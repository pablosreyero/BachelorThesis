import pandas as pd 
import numpy as np
import cv2
import copy
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

#Here we are importing the functions we created
import rpn_computation


def get_new_img_size(width, height, img_min_side=300):
	#print('Hemos entrado en get_new_image_size')
	if width <= height:
		
		f = float(img_min_side) / width
		resized_height = int(f * height)
		resized_width = img_min_side
		"""
		resized_height = 400
		resized_width = 288
		"""
	else:
		
		f = float(img_min_side) / height
		resized_width = int(f * width)
		resized_height = img_min_side
		"""
		resized_height = 288
		resized_width = 400
		"""
	return resized_width, resized_height

def augment(img_data, config, augment=True):
	#assert 'filepath' in img_data
	assert 'boxes' in img_data[1]
	assert 'w' in img_data[1]
	assert 'h' in img_data[1]
	#print('FIRSTLY WE PRINT image_data: ',img_data)
	img_data_aug = copy.deepcopy(img_data)
	 
	img = cv2.imread(img_data[0]) #We have the PATH in the 1st position of the list
	#print('Dimensions of the original image',img.shape)

	"""
	plt.imshow(img)
	plt.title("Original image")
	plt.show()
	"""

	if augment:
		rows, cols = img.shape[:2] #Preguntar a Maria José que está haciendo aqui
		if config.use_horizontal_flips and np.random.randint(0, 2) == 0: #Aqui estamos rotando las imagenes o haciendo cambios, segun el cambio que se quiera hacer se ha de especificar en el archivo de configuracion
			img = cv2.flip(img, 1)
			for bbox in img_data_aug[1]['boxes']:
				x1 = bbox['x1'] #x1 is in the position 0
				x2 = bbox['x2'] #x2 is in the position 2
				bbox['x2'] = cols - x1
				bbox['x1'] = cols - x2

		if config.use_vertical_flips and np.random.randint(0, 2) == 0:
			img = cv2.flip(img, 0)
			for bbox in img_data_aug[1]['boxes']:
				y1 = bbox['y1'] #y1 is in the position 1 
				y2 = bbox['y2'] #y2 is in the position 3 
			bbox['y2'] = rows - y1
			bbox['y1'] = rows - y2

		if config.rot_90:
			angle = np.random.choice([0,90,180,270],1)[0]
			if angle == 270:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 0)
			elif angle == 180:
				img = cv2.flip(img, -1)
			elif angle == 90:
				img = np.transpose(img, (1,0,2))
				img = cv2.flip(img, 1)
			elif angle == 0:
				pass

			for bbox in img_data_aug[1]['boxes']:
				x1 = bbox['x1']
				x2 = bbox['x2']
				y1 = bbox['y1']
				y2 = bbox['y2']
				if angle == 270:
					bbox['x1'] = y1
					bbox['x2'] = y2
					bbox['y1'] = cols - x2
					bbox['y2'] = cols - x1
				elif angle == 180:
					bbox['x2'] = cols - x1
					bbox['x1'] = cols - x2
					bbox['y2'] = rows - y1
					bbox['y1'] = rows - y2
				elif angle == 90:
					bbox['x1'] = rows - y2
					bbox['x2'] = rows - y1
					bbox['y1'] = x1
					bbox['y2'] = x2        
				elif angle == 0:
					pass

	img_data_aug[1]['w'] = img.shape[1] #Estas dos lineas me las comentó y yo las he descomentado de nuevo
	img_data_aug[1]['h'] = img.shape[0]

	#plt.imshow(img)
	#plt.title("Modified image")
	#plt.show()

	"""
	#Ahora, para asegurarnos de que nos está modificando correctamente las imágenes, las ploteamos con los BB
	box = []
	for j in img_data_aug[1]['boxes']:
		if j not in box:
			box.append(j)
	#print(box)
	#img2 = torch.tensor(img, dtype=torch.uint8)
	box = torch.tensor(box, dtype=torch.int)

	img_prime = torch.from_numpy(img) #En este paso estamos convirtiendo el np.array a tensor
	img_prime = torch.permute(img_prime, (2, 0, 1)) #Esto es para que cunado la función de bounding boxes se lea primero el número de canales.
	#print('Dimensions of the images tensor',img_prime.size())
	#print('\n')
	img2 = draw_bounding_boxes(img_prime, box, width=1, colors="red", fill=True)
						
	# transform this image to PIL image
	img2 = torchvision.transforms.ToPILImage()(img2)
    # display output

	#plt.imshow(img2)
	#plt.title("Modified image with bounding boxes")
	#plt.show()
	"""

	return img_data_aug, img


def get_anchor_gt(all_img_data, C, img_length_calc_function, mode='train'):
	""" Yield the ground-truth anchors as Y (labels)
		
	Args:
		all_img_data: list(filepath, width, height, list(bboxes))
		C: config
		img_length_calc_function: function to calculate final layer's feature map (of base model) size according to input image size
		mode: 'train' or 'test'; 'train' mode need augmentation

	Returns:
		x_img: image data after resized and scaling (smallest size = 300px)
		Y: [y_rpn_cls, y_rpn_regr]
		img_data_aug: augmented image data (original image with augmentation)
		debug_img: show image for debug
		num_pos: show number of positive anchors for debug
	"""
	while True:
		#print('QUÉ ESTÁ PASANDO AQUÍ-------------') #esto ya no lo imprime
		for img_data in all_img_data:
			print('Este es el image data detro del newsizeAugment: ',img_data)
			print(len(img_data[1]['boxes']))
			try:
				
				# read in image, and optionally add augmentation
				if mode == 'train':
					img_data_aug, x_img = augment(img_data, C, augment=True)
					#print('Augment----')
				else:
					img_data_aug, x_img = augment(img_data, C, augment=False)

				(width, height) = (img_data_aug[1]['w'], img_data_aug[1]['h']) #No entiendo porque esta haciendo esto y lo de la siguiente linea
				(rows, cols, _) = x_img.shape #que significa la barra baja?
				assert cols == width
				assert rows == height

				# get image dimensions for resizing
				(resized_width, resized_height) = get_new_img_size(width, height, C.im_size)

				# resize the image so that smalles side is length = 300px
				x_img = cv2.resize(x_img, (resized_width, resized_height), interpolation=cv2.INTER_CUBIC)
				debug_img = x_img.copy()
		
				try:
					y_rpn_cls, y_rpn_regr, num_pos = rpn_computation.calc_rpn(C, img_data_aug, width, height, resized_width, resized_height, img_length_calc_function)
					#print(y_rpn_cls,y_rpn_regr)
				except:
					print('This is the one')
					continue

				# Zero-center by mean pixel, and preprocess image
				x_img = x_img[:,:, (2, 1, 0)]  # BGR -> RGB
				x_img = x_img.astype(np.float32)
				x_img[:, :, 0] -= C.img_channel_mean[0]
				x_img[:, :, 1] -= C.img_channel_mean[1]
				x_img[:, :, 2] -= C.img_channel_mean[2]
				x_img /= C.img_scaling_factor

				x_img = np.transpose(x_img, (2, 0, 1))
				x_img = np.expand_dims(x_img, axis=0)

				y_rpn_regr[:, y_rpn_regr.shape[1]//2:, :, :] *= C.std_scaling

				x_img = np.transpose(x_img, (0, 2, 3, 1))
				y_rpn_cls = np.transpose(y_rpn_cls, (0, 2, 3, 1))
				y_rpn_regr = np.transpose(y_rpn_regr, (0, 2, 3, 1))
		
				yield np.copy(x_img), [np.copy(y_rpn_cls), np.copy(y_rpn_regr)],  img_data_aug, debug_img, num_pos
			
			except Exception as e:
				print('Es el vecino el que elige al alcalde y es el alcalde el que quiere que sean los vecinos el alcalde')
				print(e)
				#print("all_img_data",all_img_data,len(all_img_data))
				continue
			