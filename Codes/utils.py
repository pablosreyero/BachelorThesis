'''
This file contains all the utility functions, used when retrieving ground truth
data and information about some image properties
'''


from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os


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
                    final_dic[image_path]['boxes'].append({'class': 'defects' , 'x1': int(x1[iter]),'y1': int(y1[iter]),'x2': int(x2[iter]),'y2': int(y2[iter])}) #CAMBIAR TITLE por IMAGE_PATH URGENTEEEEEEEEEE
                else:
                    titlesn.append(title) #dejar la ruta desde castings
                    final_dic[image_path] = {'w': data_image1.width,
                                             'h': data_image1.height,
                                             'boxes': [{'class': 'defects', 
                                                        'x1': int(x1[iter]),
                                                        'y1': int(y1[iter]),
                                                        'x2': int(x2[iter]),
                                                        'y2': int(y2[iter])}]} 
                    #Añadir un diccionario en title 
                    #Aqui en vez de poner el titulo de cada imagen estamos poniendo la ruta de cada imagen
    if C.verbose:
        print("\n")
        print(final_dic)
        print("\n")

    #Ahora pintamos 
    for keys, stuff in final_dic.items(): #Para el append de los directorios puedo utilizar la funcion join: os.path.join
        #print(keys,stuff)

        #image_path = os.path.join(current_directory,keys) #Esta linea es muy importante
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
    print("\n")
    print("Now information of each TRAIN image will be printed")
    print("\n")

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