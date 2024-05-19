from ctypes import sizeof
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as fn
import math as mt
import os
from os import listdir
import tensorflow as tf
import pathlib as Path
import imghdr
from collections import defaultdict
from copy import deepcopy
import json
import sys
import cv2
from keras.layers import Input
from keras.models import Model
from keras.utils import generic_utils
from keras.optimizers import Adam
from keras import backend as K
import time
import random
import copy

#Here we import the used functions
import newSize_augment_anchors
import NNmodel
import layers
import losses
import rpn_to_roi
import traceback

def list_sorting(item):
    
    n = len(item)
    for i in range(n):

    # Traverse the list from 0 to n-i-1
    # (The last element will already be in place after first pass, so no need to re-check)
        for j in range(0, n-i-1):

             # Swap if current element is greater than next
            if item[j] > item[j+1]:
                item[j], item[j+1] = item[j+1],item[j]
            
    return item

def read_ground_truth(current_directory,ground_truth_name_file,sorted_images):

    image_ID= [x.split('   ')[1] for x in open('ground_truth.txt').readlines()]
    x1 = [x.split('   ')[2] for x in open('ground_truth.txt').readlines()]
    x2 = [x.split('   ')[3] for x in open('ground_truth.txt').readlines()]
    y1 = [x.split('   ')[4] for x in open('ground_truth.txt').readlines()]
    y2 = [x.split('   ')[5] for x in open('ground_truth.txt').readlines()]

    image_data = {
        "Titulos" : sorted_images,
        "ID": image_ID,
        "x1": x1,
        "x2": x2,
        "y1": y1,
        "y2": y2
    }
    return image_data

def boundingBox(current_directory,image_data):

    image_title = image_data["Titulos"]
    ID = image_data["ID"]
    x1 = image_data["x1"]
    x2 = image_data["x2"]
    y1 = image_data["y1"]
    y2 = image_data["y2"]
    y2 = [s.rstrip() for s in y2] #in order to remove the \n command at the end

    #CONVERTING FROM FLOAT TO INTEGER
    x1 = [int(float(i)) for i in x1]
    x2 = [int(float(i)) for i in x2]
    y1 = [int(float(i)) for i in y1]
    y2 = [int(float(i)) for i in y2]

    print(x1,x2,y1,y2)
    print("\n")
    print("Este es el directorio en el que tengo que trabajar; \t", current_directory)
    print("\n")
    print("Ahora probamos la implementación que queríamos poner bien")
    
    titlesn = []
    final_dic = {}
    print(image_title)
    for title in image_title:
        for iter,index in enumerate(ID): #tengo que iterar dentro del diccionario para poder coger tambien las coordenadas al mismo tiempo
            if(int(title[6:10]) == int(float(index))):
                image_path = os.path.join(current_directory,title)
                data_image1 = Image.open(image_path)
                if title in titlesn:
                    final_dic[image_path]['boxes'].append({'class': 'defects' , 'x1': int(x1[iter]),'y1': int(y1[iter]),'x2': int(x2[iter]),'y2': int(y2[iter])}) #CAMBIAR TITLE por IMAGE_PATH URGENTEEEEEEEEEE
                else:
                    titlesn.append(title) #dejar la ruta desde castings
                    final_dic[image_path] = {'w': data_image1.width,'h': data_image1.height,'boxes': [{'class': 'defects' , 'x1': int(x1[iter]),'y1': int(y1[iter]),'x2': int(x2[iter]),'y2': int(y2[iter])}]} #Añadir un diccionario en title 
                    #Aqui en vez de poner el titulo de cada imagen estamos poniendo la ruta de cada imagen
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
        #Lo que tengo que hacer es que el codigo lea el archivo .txt  del enlace que me mandó Maria José y segun vaya leyendo las imagenes que ya me dicen, el codigo tiene que saber de que imagen se trata y por ende hacer un dssplay de la información de dicha imagen

def reading_train_test (final_dic):

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
            print('\n')
            print("This are all TEST images")
            print('\n')
            print(image_title_test)
            print('\n')
        if fichiers == ('castings_train.txt'):
            image_title_train = [x for x in open('castings_train.txt').readlines()] #Aqui estamos recorriendo el archivo, MODIFICADO: image_title_train = [os.path.basename(x) for x in open('castings_train.txt').readlines()]
            image_title_train = [s.rstrip() for s in image_title_train] #Aqui le estamos quitando el simbolo de salto de linea \n
            print('\n')
            print("This are all TRAIN images")
            print('\n')
            print(image_title_train)
            print('\n')
    
    #Now that we have the titles of both train and test images that will be implemented later on, we proceed by giving the user information about these images
    print("Now information of each TEST image will be printed")
    print('\n')

    name_list1 = []
    for iter in final_dic.keys():
        name_list1.append(iter[59:])
        #name_list1.append(iter)
    print(name_list1)

    for i in image_title_test: #Como ahora ya no tengo solo los titulos de las imagenes si no que no tengo también las rutas completas de las imágenes, tengo que cambiar esta parte también
        if i in name_list1:
            i_prime = os.path.join(route_to_add,i)
            test_string = str(i_prime) + " -> " + str(final_dic[i_prime]) #MODIFICADO, ANTES: test_string = str(i) + " -> " + str(final_dic[i_prime])
            defects_test_aux = len(final_dic[i_prime]['boxes']) #i es un string
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
        if j in name_list2: #MODIFICADO, ANTES: if j in final_dict
            j_prime = os.path.join(route_to_add,j)
            train_string = [str(j_prime),(final_dic[j_prime])] #MODIFICADO, ANTES: test_string = str(j) + " -> " + str(final_dic[j_prime])
            #print(train_string)
            defects_train_aux = len(final_dic[j_prime]['boxes'])
            defects_train += defects_train_aux
            train_list.append(train_string)

    classes_count2['defects'] = defects_train

    if 'defects' not in class_mapping:
        class_mapping['defects'] = len(class_mapping)
    
    return test_list, train_list, classes_count1, classes_count2, class_mapping

def get_img_output_length(width, height):
    def get_output_length(input_length):
        return input_length//16

    return get_output_length(width), get_output_length(height)  


def main(C,output_weight_path,record_path,base_weight_path,config_output_filename):
    aux_txt = []
    # get the path/directory
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    x = []
    
    for folders in os.listdir(initial_dir):
        x.append(folders)

    #Now we sort the extracted folders
    sorted_folders = list_sorting(x)

    #Then, we iterate trough all items inside sorted_folders in order to keep folders that only host images
    for j,item in enumerate(sorted_folders):
        if item[0] != "C":
            sorted_folders.pop(j) 
    print(sorted_folders)

    merged_dictionary = {}
    for iter, chose in enumerate(sorted_folders):
        aux_img = []
        current_dir = str(initial_dir + "/" + sorted_folders[iter])
        os.chdir(current_dir) #Aqui estamos cambiando el directorio en cada iteracion para analizar cada uno de los directorios
        
        lista_de_imagenes = [ima for ima in os.listdir(current_dir) if ima.endswith(".png")]
        for images in os.listdir(current_dir):
            if (images.endswith(".png")):
                aux_img.append(images)
        
        #Aqui chequeamos que nos encontramos en el directorio correcto
        print("El directorio en el que te encuentras es el siguiente :", os.getcwd() + "\n")
        print("\n")

        buff = []
        for images in os.listdir(current_dir):
            buff.append(images) #esto me sirve para leer lo que hay en el directorio

        if ("ground_truth.txt" in buff):
            for images in os.listdir(current_dir):
                if (images.endswith(".png")):
                    aux_img.append(images)
                if (images == "ground_truth.txt"):
                    aux_txt.append(images)
                    current_directory = str(os.getcwd())

                    #A la hora de leer las imagenes del directorio C001, me sacaba la imagen 8 repetida, por ello SE ELIMINAN LOS DUPLICADOS EN LA SIGUIENTE LINEA.
                    aux_img = list(dict.fromkeys(aux_img))
                    print("\n")
                    print("Este es el aux_img:" , sorted(aux_img))

                    image_data = read_ground_truth(current_directory,images,sorted(aux_img))
                    final_dic = boundingBox(current_directory,image_data)
                    merged_dictionary = merged_dictionary | final_dic #We merge the dictionary each iteration

        else:
            print("El directorio: " + str(current_dir) + " NO contiene imágenes con defectos")
            print("\n")
            
    print("\n")
    print("Este es el diccionario final: ", merged_dictionary)
    print("\n")
    #Now that we have our dictionary with all items we search for train and test images only
    test_list, train_list, classes_count1, classes_count2, class_mapping = reading_train_test(merged_dictionary)


    print(test_list)
    print("\n")
    print(train_list)
    print("\n")
    print("Número de defectos en castings_test.txt", classes_count1)
    print("Número de defectos en castings_train.txt", classes_count2)

    print('Esto es class_mappings', class_mapping)
    print('Esto es class_count2',classes_count2)

    if 'bg' not in classes_count2:
        classes_count2['bg'] = 0
        class_mapping['bg'] = len(class_mapping)

    C.class_mapping = class_mapping

    print('Esto es classes_count2: ',classes_count2)
    print('Esto es class_mapping: ',class_mapping)

    #Now that we have all of our data extracted from .txts and images, we proceed by augmenting existing data since we are assuming an overfitting
    all_img_data = train_list
    print("Estos son los datos que nos interesan")
    print("\n")
    #print(all_img_data)

    #-------------HERE WE SHUEFFLE THE IMAGES WITH A RANDOM SEED-------------#
    random.seed(5)
    random.shuffle(all_img_data)
    
    #Now we create all anchors
    #print("Este es el all_img_data :", all_img_data)
    train_data_gen = newSize_augment_anchors.get_anchor_gt(all_img_data, C, get_img_output_length, mode='train')
    X, Y, image_data, debug_img, debug_num_pos = next(train_data_gen)


    print('Esto es el image data',image_data)
    
    #Aqui ya se empieza a pasar los datos de entreno
    print('Original image: height=%d width=%d'%(image_data[1]['h'], image_data[1]['w']))
    print('Resized image:  height=%d width=%d C.im_size=%d'%(X.shape[1], X.shape[2], C.im_size))
    print('Feature map size: height=%d width=%d C.rpn_stride=%d'%(Y[0].shape[1], Y[0].shape[2], C.rpn_stride))
    print(X.shape)
    print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
    print('Shape of y_rpn_cls {}'.format(Y[0].shape))
    print('Shape of y_rpn_regr {}'.format(Y[1].shape))
    print(image_data)

    print('Number of positive anchors for this image: %d' % (debug_num_pos))
    if debug_num_pos==0:
        gt_x1, gt_x2 = image_data[1]['boxes'][0][0]*(X.shape[2]/image_data[1]['h']), image_data[1]['boxes'][0][2]*(X.shape[2]/image_data[1]['h'])
        gt_y1, gt_y2 = image_data[1]['boxes'][0][1]*(X.shape[1]/image_data[1]['w']), image_data[1]['boxes'][0][3]*(X.shape[1]/image_data[1]['w'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

        plt.grid()
        plt.imshow(img)
        plt.show()
    else:
        cls = Y[0][0]
        pos_cls = np.where(cls==1)
        print(pos_cls)
        regr = Y[1][0]
        pos_regr = np.where(regr==1)
        print(pos_regr)
        print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0],pos_cls[1][0],:]))
        print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0],pos_regr[1][0],:]))

        gt_x1, gt_x2 = image_data[1]['boxes'][0]['x1']*(X.shape[2]/image_data[1]['w']), image_data[1]['boxes'][0]['x2']*(X.shape[2]/image_data[1]['w'])
        gt_y1, gt_y2 = image_data[1]['boxes'][0]['y1']*(X.shape[1]/image_data[1]['h']), image_data[1]['boxes'][0]['y2']*(X.shape[1]/image_data[1]['h'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        #   cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img, (gt_x1, gt_y1), (gt_x2, gt_y2), color, 2)
        cv2.circle(img, (int((gt_x1+gt_x2)/2), int((gt_y1+gt_y2)/2)), 3, color, -1)

        # Add text
        textLabel = 'gt bbox'
        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
        textOrg = (gt_x1, gt_y1+5)
        cv2.rectangle(img, (textOrg[0] - 5, textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (0, 0, 0), 2)
        cv2.rectangle(img, (textOrg[0] - 5,textOrg[1]+baseLine - 5), (textOrg[0]+retval[0] + 5, textOrg[1]-retval[1] - 5), (255, 255, 255), -1)
        cv2.putText(img, textLabel, textOrg, cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Draw positive anchors according to the y_rpn_regr
        for i in range(debug_num_pos):

            color = (100+i*(155/4), 0, 100+i*(155/4))

            idx = pos_regr[2][i*4]/4
            anchor_size = C.anchor_box_scales[int(idx/3)]
            anchor_ratio = C.anchor_box_ratios[2-int((idx+1)%3)]

            center = (pos_regr[1][i*4]*C.rpn_stride, pos_regr[0][i*4]*C.rpn_stride)
            print('Center position of positive anchor: ', center)
            cv2.circle(img, center, 3, color, -1)
            anc_w, anc_h = anchor_size*anchor_ratio[0], anchor_size*anchor_ratio[1]
            cv2.rectangle(img, (center[0]-int(anc_w/2), center[1]-int(anc_h/2)), (center[0]+int(anc_w/2), center[1]+int(anc_h/2)), color, 2)
    #         cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

    print('Green bboxes is ground-truth bbox. Others are positive anchors')
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.imshow(img)
    plt.show()

    
    #-------------------Here we're building the model-----------------------------#
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = NNmodel.nn_base(img_input, trainable=True)

    #-----------------------------------------------------------------------------#
    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
    rpn = layers.rpn_layer(shared_layers, num_anchors)

    #-----NUMBER OF CLASSES-----#
    classifier = layers.classifier_layer(shared_layers, roi_input, C.num_rois, nb_classes=len(classes_count2)) #We only have one class

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # this is a model that holds both the RPN and the classifier, used to load/save weights for the models
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # Because the google colab can only run the session several hours one time (then you need to connect again), 
    # we need to save the model and load the model to continue training
    conditional_testing = True

    if not os.path.isfile(C.model_path):
    #if conditional_testing:
        #If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            print('This is the first time of your training')
            print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')
        
        # Create the record.csv file to record losses, acc and mAP
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
        print(record_df.to_string()) #Empty data frame
    else:
        # If this is a continued training, load the trained model from before
        print('Continue training based on previous trained model')
        print('Loading weights from {}'.format(C.model_path))
        model_rpn.load_weights(C.model_path, by_name=True)
        print(model_rpn.load_weights(C.model_path, by_name=True))
        model_classifier.load_weights(C.model_path, by_name=True)
        
        # Load the records
        record_df = pd.read_csv(record_path)

        r_mean_overlapping_bboxes = record_df['mean_overlapping_bboxes']
        r_class_acc = record_df['class_acc']
        r_loss_rpn_cls = record_df['loss_rpn_cls']
        r_loss_rpn_regr = record_df['loss_rpn_regr']
        r_loss_class_cls = record_df['loss_class_cls']
        r_loss_class_regr = record_df['loss_class_regr']
        r_curr_loss = record_df['curr_loss']
        r_elapsed_time = record_df['elapsed_time']
        r_mAP = record_df['mAP']

        print('Already train %dK batches'% (len(record_df)))

#-------------------- SECOND PART OF THE TRAINING --------------------#
    optimizer = Adam(lr=1e-5)
    optimizer_classifier = Adam(lr=1e-5)
    model_rpn.compile(optimizer=optimizer, loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])

    #-------MODIFICATION-------#
    #Since we only have 1 class, we are passing 1 as the length of the class_count
    model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(len(classes_count2)-1)], metrics={'dense_class_{}'.format(len(classes_count2)): 'accuracy'})
    #model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(classes_count2-1)], metrics={'dense_class_{}'.format(classes_count): 'accuracy'})
    model_all.compile(optimizer='sgd', loss='mae')

#-------------------- TRAINING SETTING --------------------# 
    total_epochs = len(record_df)
    r_epochs = len(record_df)

    epoch_length = 150 #1000
    num_epochs = 90
    iter_num = 0

    total_epochs += num_epochs

    losses_value = np.zeros((epoch_length, 5))
    rpn_accuracy_rpn_monitor = []
    rpn_accuracy_for_epoch = []

    if len(record_df)==0:
        best_loss = np.Inf
    else:
        best_loss = np.min(r_curr_loss)       
    
    #print('length of record_df: ',len(record_df)) #result of print -> 0!
#-------------------- HERE WE'RE DELETING THE FIRST ENTRY IN THE OLD DICTIONNARY---------------
#------Becasue, the debug image is the first one in the list of dictionnaries, so if we want to compare the original input image with the result image, we have to avoid
# the first image in the old dictionnary since the first comparison to be made is with the second image

#-------------------- LAST TRAINING PART ----------------------#
    start_time = time.time()
    for epoch_num in range(num_epochs):

        progbar = generic_utils.Progbar(epoch_length)
        print('Epoch {}/{}'.format(r_epochs + 1, total_epochs))
        
        r_epochs += 1

        aa = 0 #This is for the final dictionnary all_images, in order to plot the original input image with its corresponding bounding boxes

        while True:
            try:

                if len(rpn_accuracy_rpn_monitor) == epoch_length and C.verbose:
                    mean_overlapping_bboxes = float(sum(rpn_accuracy_rpn_monitor))/len(rpn_accuracy_rpn_monitor)
                    rpn_accuracy_rpn_monitor = []
    #                 print('Average number of overlapping bounding boxes from RPN = {} for {} previous iterations'.format(mean_overlapping_bboxes, epoch_length))
                    if mean_overlapping_bboxes == 0:
                        print('RPN is not producing bounding boxes that overlap the ground truth boxes. Check RPN settings or keep training.')

                X, Y, img_data, debug_img, debug_num_pos = next(train_data_gen)
                # Train rpn model and get loss value [_, loss_rpn_cls, loss_rpn_regr]
                #-------------------A PARTIR DE AQUI TERMINAMOS-----------------------#

                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)
                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                #print('Tamaño P_rpn[0] y P_rpn[1]', P_rpn[0].shape,P_rpn[1].shape)
                #R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(), use_regr=True, overlap_thresh=0.7, max_boxes=300)
                R = rpn_to_roi.rpn_to_roi(P_rpn[0], P_rpn[1], C, K.set_image_data_format('channels_last'), use_regr=True,  max_boxes=50, overlap_thresh=0.4) #Due to an update in keras library, image_dim_ordering() ---> set_image_data_format('channels_last')
                #Here I make a deep copy of R in order to further convert R's type
                
                R2 = copy.deepcopy(R) 
                R2 = R2.tolist()
                
                #print('\nR2 ORIGINAL: ', R2)

                #Ahora multiplicamos todos los elementos del feature map por 16
                for i in R2:
                    for posi, objeto in enumerate(i):
                        i[posi] = C.rpn_stride * objeto
                
                #print('\nNUEVO R2: ',R2)
                
                X_prime = np.transpose(X,(2,1,3,0))
                #print('Tamaño de X_PRIME, después del reshape: ',X_prime.shape)
                X_prime = np.squeeze(X_prime)
                X_prime = X_prime.astype(dtype=np.uint8)
                
                #print('Tamaño final de X_prime: ', X_prime.shape)
                #print('ESTE ES EL TIPO DE VARIABLE DE X_prime: ', type(X_prime))
                #print('CONTENIDO DE X_prime: ', X_prime)

                #----Aqui sacamos los BB del image data para pintar los BB encima de la imágen original-----
            
                #print('\nESTE ES EL IMG_DATA por si a caso: ', img_data)
                #print('ESTE ES EL ALL_IMG_DATA por si a caso: ', all_img_data[aa+1])
                
                boxBB = []
                boxBB2 = []
                for j in all_img_data[aa+1][1]['boxes']:
                    for k in j:
                        if k != 'class':
                            boxBB.append(j[k])                        
                    boxBB2.append(copy.deepcopy(boxBB))
                    boxBB.clear()

                img = read_image(all_img_data[aa+1][0])
                #print('Esta es la imagen que estamos leyendo: ',all_img_data[aa+1][0])
                aa+=1
                boxBB2 = torch.tensor(boxBB2, dtype=torch.int)
                img = draw_bounding_boxes(img, boxBB2, width=1, colors='red', fill=True)
                                    
                # transform this image to PIL image
                img = torchvision.transforms.ToPILImage()(img)
            
                #----Here we extract anchors in order to plot them on the processed image by the NN---------
                xi = 'Imagen final'
                
                imagex = np.ascontiguousarray(X_prime, dtype=np.uint8)
                #print('image.shape: ',)

                #-------------REPRESENTACION CON EL CODIGO DE ARRIBA-------------
                color = (255,0,0)
                boxx = []
                for j in R2:
                    if j not in boxx:
                        boxx.append(j)
                        cv2.rectangle(imagex, (j[0],j[1]), (j[0]+j[2], j[1]+j[3]), color, 2)
        

                rows, cols = 1, 2
                plt.subplot(rows, cols, 1)
                plt.imshow(img)
                plt.title('Imagen ORIGINAL')

                plt.subplot(rows, cols, 2)
                plt.imshow(imagex)
                plt.title(xi)
                
                plt.show()
            
                
                #'channels_last' for tensorflow, 'channels_first' for Theano and 'channels_last' for CNTK (Microsoft Cognitive Toolkit)
                
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h) format
                # X2: bboxes that iou > C.classifier_min_overlap for all gt bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes

                X2, Y1, Y2, IouS = losses.calc_iou(R, img_data, C, class_mapping)       

                # If X2 is None means there are no matching bboxes
                if X2 is None:
                    rpn_accuracy_rpn_monitor.append(0)
                    rpn_accuracy_for_epoch.append(0)
                    continue
                
                # Find out the positive anchors and negative anchors
                neg_samples = np.where(Y1[0, :, -1] == 1)
                pos_samples = np.where(Y1[0, :, -1] == 0)

                if len(neg_samples) > 0:
                    neg_samples = neg_samples[0]
                else:
                    neg_samples = []

                if len(pos_samples) > 0:
                    pos_samples = pos_samples[0]
                else:
                    pos_samples = []
                #print('D')

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples, C.num_rois//2, replace=False).tolist()
                    
                    # Randomly choose (num_rois - num_pos) neg samples
                    try:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=False).tolist()
                    except:
                        selected_neg_samples = np.random.choice(neg_samples, C.num_rois - len(selected_pos_samples), replace=True).tolist()
                    
                    # Save all the pos and neg samples in sel_samples
                    sel_samples = selected_pos_samples + selected_neg_samples
                else:
                    # in the extreme case where num_rois = 1, we pick a random pos or neg sample
                    selected_pos_samples = pos_samples.tolist()
                    selected_neg_samples = neg_samples.tolist()
                    if np.random.randint(0, 2):
                        sel_samples = random.choice(neg_samples)
                    else:
                        sel_samples = random.choice(pos_samples)
                
                # training_data: [X, X2[:, sel_samples, :]]
                # labels: [Y1[:, sel_samples, :], Y2[:, sel_samples, :]]
                #  X                     => img_data resized image
                #  X2[:, sel_samples, :] => num_rois (4 in here) bboxes which contains selected neg and pos
                #  Y1[:, sel_samples, :] => one hot encode for num_rois bboxes which contains selected neg and pos
                #  Y2[:, sel_samples, :] => labels and gt bboxes for num_rois bboxes which contains selected neg and pos
                loss_class = model_classifier.train_on_batch([X, X2[:, sel_samples, :]], [Y1[:, sel_samples, :], Y2[:, sel_samples, :]])
                losses_value[iter_num, 0] = loss_rpn[1]
                losses_value[iter_num, 1] = loss_rpn[2]

                losses_value[iter_num, 2] = loss_class[1]
                losses_value[iter_num, 3] = loss_class[2]
                losses_value[iter_num, 4] = loss_class[3]

                iter_num += 1

                progbar.update(iter_num, [('rpn_cls', np.mean(losses_value[:iter_num, 0])), ('rpn_regr', np.mean(losses_value[:iter_num, 1])),
                                        ('final_cls', np.mean(losses_value[:iter_num, 2])), ('final_regr', np.mean(losses_value[:iter_num, 3]))])
                
                #print('Esto es iter_num y epoch_length: ',iter_num,epoch_length)
                if iter_num == epoch_length:
                    loss_rpn_cls = np.mean(losses_value[:, 0])
                    loss_rpn_regr = np.mean(losses_value[:, 1])
                    loss_class_cls = np.mean(losses_value[:, 2])
                    loss_class_regr = np.mean(losses_value[:, 3])
                    class_acc = np.mean(losses_value[:, 4])

                    mean_overlapping_bboxes = float(sum(rpn_accuracy_for_epoch)) / len(rpn_accuracy_for_epoch)
                    rpn_accuracy_for_epoch = []

                    if C.verbose:
                        print('Mean number of bounding boxes from RPN overlapping ground truth boxes: {}'.format(mean_overlapping_bboxes))
                        print('Classifier accuracy for bounding boxes from RPN: {}'.format(class_acc))
                        print('Loss RPN classifier: {}'.format(loss_rpn_cls))
                        print('Loss RPN regression: {}'.format(loss_rpn_regr))
                        print('Loss Detector classifier: {}'.format(loss_class_cls))
                        print('Loss Detector regression: {}'.format(loss_class_regr))
                        print('Total loss: {}'.format(loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr))
                        print('Elapsed time: {}'.format(time.time() - start_time))
                        elapsed_time = (time.time()-start_time)/60

                    curr_loss = loss_rpn_cls + loss_rpn_regr + loss_class_cls + loss_class_regr
                    iter_num = 0
                    start_time = time.time()

                    if curr_loss < best_loss:
                        if C.verbose:
                            print('Total loss decreased from {} to {}, saving weights'.format(best_loss,curr_loss))
                        best_loss = curr_loss
                        model_all.save_weights(C.model_path)

                    new_row = {'mean_overlapping_bboxes':round(mean_overlapping_bboxes, 3), 
                            'class_acc':round(class_acc, 3), 
                            'loss_rpn_cls':round(loss_rpn_cls, 3), 
                            'loss_rpn_regr':round(loss_rpn_regr, 3), 
                            'loss_class_cls':round(loss_class_cls, 3), 
                            'loss_class_regr':round(loss_class_regr, 3), 
                            'curr_loss':round(curr_loss, 3), 
                            'elapsed_time':round(elapsed_time, 3), 
                            'mAP': 0}

                    record_df = record_df.append(new_row, ignore_index=True)
                    record_df.to_csv(record_path, index=0)

                    break

            except Exception as e:
                print('------------------------EXCEPCIÓN------------------------ \n')
                print('Exception: {}'.format(e))
                traceback.print_stack()
                print('--------------------------------------------------------- \n')
                #exit()
                continue

    print('Training complete, exiting.')
    
    #-------------------HERE WE ARE PLOTTING ALL RESULTS TO BETTER SEE RESULTS----------------#
    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['mean_overlapping_bboxes'], 'r')
    plt.title('mean_overlapping_bboxes')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['class_acc'], 'r')
    plt.title('class_acc')

    plt.show()

    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'r')
    plt.title('loss_rpn_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'r')
    plt.title('loss_rpn_regr')
    plt.show()


    plt.figure(figsize=(15,5))
    plt.subplot(1,2,1)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    plt.title('loss_class_cls')
    plt.subplot(1,2,2)
    plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'r')
    plt.title('loss_class_regr')
    plt.show()

    plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    plt.title('total_loss')
    plt.show()
    
    # plt.figure(figsize=(15,5))
    # plt.subplot(1,2,1)
    # plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'r')
    # plt.title('total_loss')
    # plt.subplot(1,2,2)
    # plt.plot(np.arange(0, r_epochs), record_df['elapsed_time'], 'r')
    # plt.title('elapsed_time')
    # plt.show()

    # plt.title('loss')
    # plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_cls'], 'b')
    # plt.plot(np.arange(0, r_epochs), record_df['loss_rpn_regr'], 'g')
    # plt.plot(np.arange(0, r_epochs), record_df['loss_class_cls'], 'r')
    # plt.plot(np.arange(0, r_epochs), record_df['loss_class_regr'], 'c')
    # # plt.plot(np.arange(0, r_epochs), record_df['curr_loss'], 'm')
    # plt.show()


    
    