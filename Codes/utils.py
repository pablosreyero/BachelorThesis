'''
This file contains all the utility functions, used when retrieving ground truth
data and information about some image properties
'''

import cv2
import glob
import numpy as np
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
import copy


def list_sorting(item):
    '''
    In this function we are sorting the elements of the input list.
    Input:
    - item (python list)
    Output:
    - item (python list)
    '''

    n = len(item)
    for i in range(n):

    # Traverse the list from 0 to n-i-1
    # (The last element will already be in place after first pass,
    # so no need to re-check)

        for j in range(0, n-i-1):
             # Swap if current element is greater than next
            if item[j] > item[j+1]:
                item[j], item[j+1] = item[j+1],item[j]
    return item


def read_ground_truth(sorted_images):
    '''
    As the name implies, we are reading all the information from the
    groundtruth file.
    Inputs:
    - sorted_images (str) contains all the image titles for all the images in
    the repository.
    Outputs:
    - image_data (python dictionary) contains all the information about the
    groundtruth coordinates.
    '''

    image_ID= [x.split('   ')[1] for x in open('ground_truth.txt').readlines()]
    x1 = [x.split('   ')[2] for x in open('ground_truth.txt').readlines()]
    x2 = [x.split('   ')[3] for x in open('ground_truth.txt').readlines()]
    y1 = [x.split('   ')[4] for x in open('ground_truth.txt').readlines()]
    y2 = [x.split('   ')[5] for x in open('ground_truth.txt').readlines()]

    image_data = {"Titulos" : sorted_images,
                  "ID": image_ID,
                  "x1": x1,
                  "x2": x2,
                  "y1": y1,
                  "y2": y2}
    return image_data


def boundingBox(C,current_directory,image_data):
    '''
    We are now retrieving all the information about the BB to plot them.
    Input:
    - C (python class) this is the configuration class.
    - current_directory (python str) contains the path of the current
    directory.
    - image_data (python dict) contains all the defect's coordinates of each
    training image.
    Output:
    - final_dic (python dict) contains all the information about each training
    image.
    '''

    image_title = image_data["Titulos"]
    ID = image_data["ID"]
    x1 = image_data["x1"]
    x2 = image_data["x2"]
    y1 = image_data["y1"]
    y2 = image_data["y2"]
    y2 = [s.rstrip() for s in y2] #in order to remove the \n command at the end

    x1 = [int(float(i)) for i in x1]
    x2 = [int(float(i)) for i in x2]
    y1 = [int(float(i)) for i in y1]
    y2 = [int(float(i)) for i in y2]

    if C.verbose:
        print(x1,x2,y1,y2)
        print("\n")
        print("Este es el directorio en el que tengo que trabajar; \t", current_directory)
        print("\n")
        print("Ahora probamos la implementación que queríamos poner bien")

    titlesn = []
    final_dic = {}
    if C.verbose: print(image_title)
    for title in image_title:
        for iter,index in enumerate(ID): # tengo que iterar dentro del diccionario para poder coger tambien las coordenadas al mismo tiempo
            if(int(title[6:10]) == int(float(index))):
                image_path = os.path.join(current_directory,title)
                data_image1 = Image.open(image_path)
                if title in titlesn:
                    final_dic[image_path]['boxes'].append({'class': 'defects',
                                                           'x1': int(x1[iter]),
                                                           'y1': int(y1[iter]),
                                                           'x2': int(x2[iter]),
                                                           'y2': int(y2[iter])})
                else:
                    titlesn.append(title) #dejar la ruta desde castings
                    final_dic[image_path] = {'w': data_image1.width,
                                             'h': data_image1.height,
                                             'boxes': [{'class': 'defects', 
                                                        'x1': int(x1[iter]),
                                                        'y1': int(y1[iter]),
                                                        'x2': int(x2[iter]),
                                                        'y2': int(y2[iter])}]} 
    if C.verbose: print(f"\n{final_dic}\n")

    #Ahora pintamos 
    for keys, stuff in final_dic.items(): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        #print(keys,stuff)

        #Esta linea es muy importante
        #image_path = os.path.join(current_directory,keys)
        #print("\n")
        img = read_image(keys)
        box = []
        box11 = []
        box12 = []
        for j in stuff['boxes']:
            for k in range(len(final_dic[keys])):
                if j not in box:
                    box.append(j)
                    box11 = list(j.values())
                    box11.pop(0)
                    box12.append(box11)

        box12 = torch.tensor(box12, dtype=torch.int)
        img = draw_bounding_boxes(img, box12, width=1, colors='red', fill=True)
                  
        # transform this image to PIL image
        img = torchvision.transforms.ToPILImage()(img)

        #img.show()

    return final_dic
        #Lo que tengo que hacer es que el codigo lea el archivo .txt  del
        # enlace que me mandó Maria José y segun vaya leyendo las imagenes que
        # ya me dicen, el codigo tiene que saber de que imagen se trata y por
        # ende hacer un display de la información de dicha imagen


def reading_train_test (C,final_dic):
    '''
    In this file we are going through the training and test images, and we
    craft a training and testing list.

    Input:
    - C (python class) this is the configuration class.
    - final_dict (python dict)
    '''

    train_list = []
    test_list = []
    classes_count1 = {}
    classes_count2 = {}
    class_mapping = {}
    defects_test = 0
    defects_train = 0

    route = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/Ferguson/metadata/gdxray"
    route_to_add = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/"

    os.chdir(route)
    for fichiers in os.listdir(route):
        print(fichiers)
        if fichiers == ('castings_test.txt'):
            image_title_test = [x for x in open('castings_test.txt').readlines()] #Aqui estamos recorriendo el archivo MODIFICADO, ANTES; image_title_test = [os.path.basename(x) for x in open('castings_test.txt').readlines()]
            image_title_test = [s.rstrip() for s in image_title_test] #Aqui le estamos quitando el simbolo de salto de linea \n

            if C.verbose:
                print('\n')
                print("This are all TEST images")
                print('\n')
                print(image_title_test)
                print('\n')

        if fichiers == ('castings_train.txt'):
            image_title_train = [x for x in open('castings_train.txt').readlines()] #Aqui estamos recorriendo el archivo, MODIFICADO: image_title_train = [os.path.basename(x) for x in open('castings_train.txt').readlines()]
            image_title_train = [s.rstrip() for s in image_title_train] #Aqui le estamos quitando el simbolo de salto de linea \n

            if C.verbose:
                print('\n')
                print("This are all TRAIN images")
                print('\n')
                print(image_title_train)
                print('\n')

    name_list1 = []
    for iter in final_dic.keys():
        name_list1.append(iter[59:])
        #name_list1.append(iter)
    if C.verbose: print(name_list1)

    for i in image_title_test: # Como ahora ya no tengo solo los titulos de las imagenes si no que no tengo también las rutas completas de las imágenes, tengo que cambiar esta parte también
        if i in name_list1:
            i_prime = os.path.join(route_to_add,i)
            test_string = str(i_prime) + " -> " + str(final_dic[i_prime]) # MODIFICADO, ANTES: test_string = str(i) + " -> " + str(final_dic[i_prime])
            defects_test_aux = len(final_dic[i_prime]['boxes']) # i es un string
            defects_test += defects_test_aux
            test_list.append(test_string)

    classes_count1['defects'] = defects_test
    if C.verbose: print("\nNow information of each TRAIN image is printed\n")

    name_list2 = []
    for iter in final_dic.keys():
        name_list2.append(iter[59:])

    for j in image_title_train:
        if j in name_list2: # MODIFICADO, ANTES: if j in final_dict
            j_prime = os.path.join(route_to_add,j)
            train_string = [str(j_prime),(final_dic[j_prime])] # MODIFICADO, ANTES: test_string = str(j) + " -> " + str(final_dic[j_prime])
            defects_train_aux = len(final_dic[j_prime]['boxes'])
            defects_train += defects_train_aux
            train_list.append(train_string)

    classes_count2['defects'] = defects_train

    if 'defects' not in class_mapping:
        class_mapping['defects'] = len(class_mapping)

    return test_list, train_list, classes_count1, classes_count2, class_mapping


def get_img_output_length(width, height):
    '''
    With this function we are getting both the width and the height from each
    image

    Inputs:
    - width (python int)
    - height (python int)

    Outputs:
    - width (python int)
    - height (python int)
    '''

    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)


def calculate_channel_means(all_image_data):
    """
    This function takes as an input all the images paths, and returns the total
    img_channel_mean computed from each image in the training set.
    """

    # Initialize sums for each channel
    sum_r, sum_g, sum_b = 0, 0, 0
    num_pixels = 0
    count = 0
    for image_path in all_image_data:
        img = cv2.imread(image_path[0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        sum_r += np.sum(img[:, :, 0])
        sum_g += np.sum(img[:, :, 1])
        sum_b += np.sum(img[:, :, 2])
        num_pixels += img.shape[0] * img.shape[1]
        count += 1

    mean_r = sum_r / num_pixels
    mean_g = sum_g / num_pixels
    mean_b = sum_b / num_pixels
    print(f"{count} analysed images and computed means")

    return [mean_r, mean_g, mean_b]


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