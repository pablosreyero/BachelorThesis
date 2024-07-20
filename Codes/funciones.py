import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
import os
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
from rpn_computation import get_anchor_gt
import NNmodel
import layers
import losses
import rpn_to_roi
import traceback
import utils


def main(C,output_weight_path,record_path,base_weight_path,config_output_filename):
    aux_txt = []
    # get the path/directory
    initial_dir = "/Users/pablosreyero/Documents/Universidad/TFG/tfg-psr/data/Castings"
    x = []

    for folders in os.listdir(initial_dir):
        x.append(folders)

    #Now we sort the extracted folders
    sorted_folders = utils.list_sorting(x)

    '''Iterate trough all items inside sorted_folders to keep folders that only
    # host images'''

    for j,item in enumerate(sorted_folders):
        if item[0] != "C":
            sorted_folders.pop(j)

    merged_dictionary = {}
    for iter, _ in enumerate(sorted_folders):
        aux_img = []
        current_dir = str(initial_dir + "/" + sorted_folders[iter])
        os.chdir(current_dir) # Change folder in each it to analize all folders
        # lista_de_imagenes = [ima for ima in os.listdir(current_dir) if ima.endswith(".png")]
        for images in os.listdir(current_dir):
            if images.endswith(".png"):
                aux_img.append(images)

        # Check if we are at the correct folder
        if C.verbose: print(f"You are in: {os.getcwd()}\n")

        buff = []
        for images in os.listdir(current_dir):
            buff.append(images) # read inside directory

        if "ground_truth.txt" in buff:
            for images in os.listdir(current_dir):
                if images.endswith(".png"):
                    aux_img.append(images)
                if images == "ground_truth.txt":
                    aux_txt.append(images)
                    current_directory = str(os.getcwd())

                    # When reading images from C001 dir, image 8 was repeating
                    # therefore in the following lines we are deleting clones
                    aux_img = list(dict.fromkeys(aux_img))
                    if C.verbose:
                        print(f"Este es el aux_img: {sorted(aux_img)}")

                    image_data = utils.read_ground_truth(sorted(aux_img))
                    final_dic = utils.boundingBox(C,current_directory,
                                                  image_data)

                    # We merge the dictionary each iteration
                    merged_dictionary = merged_dictionary | final_dic
        else:
            if C.verbose: print(f"{str(current_dir)} has NO deffects\n")

    if C.verbose: print(f"\nfinal DICT: {merged_dictionary}\n")

    # Dictionary is ready, NOW search for train and test images only
    results_reading = utils.reading_train_test(C,merged_dictionary)

    test_lst = results_reading[0]
    train_lst = results_reading[1]
    cls_count1 = results_reading[2]
    cls_count2 = results_reading[3]
    cls_map = results_reading[4]

    if C.verbose: print(f"\n{test_lst}\n")
    if C.verbose: print(f"\n{train_lst}\n")
    if C.verbose: print("Número de defectos en castings_test.txt", cls_count1)
    if C.verbose: print("Número de defectos en castings_train.txt", cls_count2)
    if C.verbose: print('Esto es cls_maps', cls_map)
    if C.verbose: print('Esto es class_count2',cls_count2)

    if 'bg' not in cls_count2:
        cls_count2['bg'] = 0
        cls_map['bg'] = len(cls_map)

    C.class_mapping = cls_map

    if C.verbose: print(f'This is cls_count2: {cls_count2}')
    if C.verbose: print(f'Esto es cls_map: {cls_map}')

    # Data from images extracted! -> NOW augment data (assume overfitting)
    all_img_data = train_lst

    #-------------HERE WE SHUEFFLE THE IMAGES WITH A RANDOM SEED-------------#
    random.seed(5)
    random.shuffle(all_img_data)

    #Now we create all anchors
    if C.verbose: print("Este es el all_img_data :", all_img_data)

    # now let us compute all channel_means form all images

    if C.verbose: print("Computing channel means of all images")
    channel_means = utils.calculate_channel_means(all_img_data)
    C.img_channel_mean = channel_means
    if C.verbose: print(f"Channel_mean: {C.img_channel_mean}")

    train_data_gen = get_anchor_gt(all_img_data,
                                   C,
                                   utils.get_img_output_length,
                                   mode='train')

    X, Y, image_data, debug_img, debug_num_pos = next(train_data_gen)


    if C.verbose: print(f'Esto es el image data: {image_data}')

    #Aqui ya se empieza a pasar los datos de entreno
    if C.verbose:
        print('Original image: height=%d width=%d'%(image_data[1]['h'],
                                                    image_data[1]['w']))
        print('Resized image:  height=%d width=%d C.im_size=%d'%(X.shape[1],
                                                                 X.shape[2],
                                                                 C.im_size))
        print('Feature map size: height=%d width=%d C.rpn_stride=%d'%(Y[0].shape[1],
                                                                      Y[0].shape[2],
                                                                      C.rpn_stride))
        print(X.shape)
        print(str(len(Y))+" includes 'y_rpn_cls' and 'y_rpn_regr'")
        print('Shape of y_rpn_cls {}'.format(Y[0].shape))
        print('Shape of y_rpn_regr {}'.format(Y[1].shape))
        print(image_data)
        print('Number of positive anchors for this image: %d' % (debug_num_pos))

    if debug_num_pos==0:
        print("PRINTING the DEBUG image")
        gt_x1, gt_x2 = image_data[1]['boxes'][0][0]*(X.shape[2]/image_data[1]['h']), image_data[1]['boxes'][0][2]*(X.shape[2]/image_data[1]['h'])
        gt_y1, gt_y2 = image_data[1]['boxes'][0][1]*(X.shape[1]/image_data[1]['w']), image_data[1]['boxes'][0][3]*(X.shape[1]/image_data[1]['w'])
        gt_x1,gt_y1, gt_x2, gt_y2 = int(gt_x1),int(gt_y1),int(gt_x2),int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)

        cv2.putText(img,
                    'gt bbox',
                    (gt_x1, gt_y1-5),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.7,
                    color,
                    1)
        cv2.rectangle(img,
                      (gt_x1, gt_y1),
                      (gt_x2, gt_y2),
                      color,
                      2)
        cv2.circle(img,
                   (int((gt_x1+gt_x2)/2),int((gt_y1+gt_y2)/2)),
                   3,
                   color,
                   -1)

        plt.grid()
        plt.imshow(img)
        plt.show()

    else:
        print(f"PRINTING the DEBUG image (else statement) & debug_num_pos = {debug_num_pos}")
        cls = Y[0][0]
        pos_cls = np.where(cls==1)
        if C.verbose: print(pos_cls)
        regr = Y[1][0]
        pos_regr = np.where(regr==1)
        if C.verbose:
            print(pos_regr)
            print('y_rpn_cls for possible pos anchor: {}'.format(cls[pos_cls[0][0],pos_cls[1][0],:]))
            print('y_rpn_regr for positive anchor: {}'.format(regr[pos_regr[0][0],pos_regr[1][0],:]))

        gt_x1, gt_x2 = image_data[1]['boxes'][0]['x1']*(X.shape[2]/image_data[1]['w']), image_data[1]['boxes'][0]['x2']*(X.shape[2]/image_data[1]['w'])
        gt_y1, gt_y2 = image_data[1]['boxes'][0]['y1']*(X.shape[1]/image_data[1]['h']), image_data[1]['boxes'][0]['y2']*(X.shape[1]/image_data[1]['h'])
        gt_x1, gt_y1, gt_x2, gt_y2 = int(gt_x1), int(gt_y1), int(gt_x2), int(gt_y2)

        img = debug_img.copy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        color = (0, 255, 0)
        # cv2.putText(img, 'gt bbox', (gt_x1, gt_y1-5), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
        cv2.rectangle(img,
                      (gt_x1, gt_y1),
                      (gt_x2, gt_y2),
                      color,
                      2)
        cv2.circle(img,
                   (int((gt_x1+gt_x2)/2),
                    int((gt_y1+gt_y2)/2)),
                    3,
                    color,
                    -1)

        # Add text
        textLabel = 'gt bbox'
        (retval,baseLine) = cv2.getTextSize(textLabel,cv2.FONT_HERSHEY_COMPLEX,0.5,1)
        textOrg = (gt_x1, gt_y1+5)

        cv2.rectangle(img,
                      (textOrg[0] - 5,
                       textOrg[1]+baseLine - 5),
                       (textOrg[0]+retval[0] + 5,
                        textOrg[1]-retval[1] - 5),
                        (0, 0, 0), 2)
        cv2.rectangle(img,
                      (textOrg[0] - 5,
                       textOrg[1]+baseLine - 5),
                       (textOrg[0]+retval[0] + 5,
                        textOrg[1]-retval[1] - 5),
                        (255, 255, 255), -1)
        cv2.putText(img,
                    textLabel,
                    textOrg,
                    cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 0, 0), 1)

        # Draw positive anchors according to the y_rpn_regr
        for i in range(debug_num_pos):
            color = (100+i*(155/4), 0, 100+i*(155/4))
            idx = pos_regr[2][i*4]/4
            anchor_size = C.anchor_box_scales[int(idx/3)]
            anchor_ratio = C.anchor_box_ratios[2-int((idx+1)%3)]
            center = (pos_regr[1][i*4]*C.rpn_stride,
                      pos_regr[0][i*4]*C.rpn_stride)

            print('Center position of positive anchor: ', center)
            cv2.circle(img, center, 3, color, -1)
            anc_w, anc_h = anchor_size*anchor_ratio[0],anchor_size*anchor_ratio[1]
            cv2.rectangle(img,
                          (center[0]-int(anc_w/2),
                           center[1]-int(anc_h/2)),
                           (center[0]+int(anc_w/2),
                            center[1]+int(anc_h/2)),
                            color,
                            2)
    # cv2.putText(img, 'pos anchor bbox '+str(i+1), (center[0]-int(anc_w/2), center[1]-int(anc_h/2)-5), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    print('GREEN bboxes ground-truth. other -> positive anchors')
    plt.figure(figsize=(8,8))
    plt.grid()
    plt.imshow(img)
    plt.show()

    '''
    #-------------------Here we're building the model-------------------#
    input_shape_img = (None, None, 3)

    img_input = Input(shape=input_shape_img)
    roi_input = Input(shape=(None, 4))

    # define the base network (VGG here, can be Resnet50, Inception, etc)
    shared_layers = NNmodel.nn_base(img_input, trainable=True)

    # define the RPN, built on the base layers
    num_anchors = len(C.anchor_box_scales) * len(C.anchor_box_ratios) # 9
    rpn = layers.rpn_layer(shared_layers, num_anchors)

    #-----NUMBER OF CLASSES-----#
    classifier = layers.classifier_layer(shared_layers,
                                         roi_input,
                                         C.num_rois,
                                         nb_classes=len(cls_count2)) 
                                        # We only have one class

    model_rpn = Model(img_input, rpn[:2])
    model_classifier = Model([img_input, roi_input], classifier)

    # This holds both the RPN and classifier -> load/save weights for the model
    model_all = Model([img_input, roi_input], rpn[:2] + classifier)

    # Because the google colab can only run the session several hours one time 
    # (then you need to connect again), we need to save the model and load it
    # to continue training
    conditional_testing = True

    if not os.path.isfile(C.model_path):
    #if conditional_testing:
        #If this is the begin of the training, load the pre-traind base network such as vgg-16
        try:
            if C.verbose: print('This is the first time of your training')
            if C.verbose: print('loading weights from {}'.format(C.base_net_weights))
            model_rpn.load_weights(C.base_net_weights, by_name=True)
            model_classifier.load_weights(C.base_net_weights, by_name=True)
        except:
            if C.verbose: print('Could not load pretrained model weights. Weights can be found in the keras application folder \
                https://github.com/fchollet/keras/tree/master/keras/applications')
        
        # Create the record.csv file to record losses, acc and mAP
        record_df = pd.DataFrame(columns=['mean_overlapping_bboxes', 'class_acc', 'loss_rpn_cls', 'loss_rpn_regr', 'loss_class_cls', 'loss_class_regr', 'curr_loss', 'elapsed_time', 'mAP'])
        if C.verbose: print(record_df.to_string()) #Empty data frame
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

        print('Already trained %dK batches'% (len(record_df)))

#-------------------- SECOND PART OF THE TRAINING --------------------#
    optimizer = Adam(learning_rate=C.learning_rate) # the original lr was 1e-5
    optimizer_classifier = Adam(learning_rate=C.learning_rate)
    model_rpn.compile(optimizer=optimizer,
                      loss=[losses.rpn_loss_cls(num_anchors),
                            losses.rpn_loss_regr(num_anchors)])

    #-------MODIFICATION-------#
    #Since we only have 1 class, we are passing 1 as the length of the class_count
    model_classifier.compile(optimizer=optimizer_classifier,
                             loss=[losses.class_loss_cls,
                                   losses.class_loss_regr(len(cls_count2)-1)],
                                   metrics={'dense_class_{}'.format(len(cls_count2)): 'accuracy'})
    #model_classifier.compile(optimizer=optimizer_classifier, loss=[losses.class_loss_cls, losses.class_loss_regr(cls_count2-1)], metrics={'dense_class_{}'.format(classes_count): 'accuracy'})
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
#-------------------- HERE WE'RE DELETING THE FIRST ENTRY IN THE OLD DICTIONARY---------------
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
                # Train rpn model and get loss value [_, loss_rpn_cls,loss_rpn_regr]
                #------------A PARTIR DE AQUI TERMINAMOS------------#
                loss_rpn = model_rpn.train_on_batch(X, Y)

                # Get predicted rpn from rpn model [rpn_cls, rpn_regr]
                P_rpn = model_rpn.predict_on_batch(X)
                # R: bboxes (shape=(300,4))
                # Convert rpn layer to roi bboxes
                # print('Tamaño P_rpn[0] y P_rpn[1]', P_rpn[0].shape,P_rpn[1].shape)
                # R = rpn_to_roi(P_rpn[0], P_rpn[1], C, K.image_dim_ordering(),
                # use_regr=True, overlap_thresh=0.7, max_boxes=300)
                R = rpn_to_roi.rpn_to_roi(P_rpn[0],
                                          P_rpn[1],
                                          C,
                                          K.set_image_data_format('channels_last'),
                                          use_regr=True,
                                          max_boxes=50,
                                          overlap_thresh=0.6) # the overlap thresh was 0.4
    
                # Due to an update in keras library, image_dim_ordering()--->
                # set_image_data_format('channels_last')
                # Here I make a deep copy of R in order to further convert R's type
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

                #----Aqui sacamos los BB del image data para pintar los BB
                # encima de la imágen original-----
            
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
                img = draw_bounding_boxes(img,
                                          boxBB2,
                                          width=1,
                                          colors='red',
                                          fill=True)
                                    
                # transform this image to PIL image
                img = torchvision.transforms.ToPILImage()(img)
            
                #----Here we extract anchors in order to plot them on the processed image by the NN---------
                imagex = np.ascontiguousarray(X_prime, dtype=np.uint8)
                #print('image.shape: ',)

                #-------------REPRESENTACION CON EL CODIGO DE ARRIBA-------------
                color = (255,0,0) # the red color of boxes
                boxx = []
                for j in R2:
                    if j not in boxx:
                        boxx.append(j)
                        cv2.rectangle(imagex,
                                      (j[0],
                                       j[1]),
                                       (j[0]+j[2],
                                        j[1]+j[3]),
                                        color,
                                        2)

                rows, cols = 1, 2
                plt.subplot(rows, cols, 1)
                plt.imshow(img)
                plt.title('Imagen ORIGINAL')

                plt.subplot(rows, cols, 2)
                plt.imshow(imagex)
                plt.title('Imagen FINAL')
                
                plt.show()
                #'channels_last' for tensorflow, 'channels_first' for Theano
                # and 'channels_last' for CNTK (Microsoft Cognitive Toolkit)
                
                # note: calc_iou converts from (x1,y1,x2,y2) to (x,y,w,h)format
                # X2: bboxes that iou > C.classifier_min_overlap for all gt
                # bboxes in 300 non_max_suppression bboxes
                # Y1: one hot code for bboxes from above => x_roi (X)
                # Y2: corresponding labels and corresponding gt bboxes

                X2, Y1, Y2, IouS = losses.calc_iou(R, img_data, C, cls_map)       

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

                rpn_accuracy_rpn_monitor.append(len(pos_samples))
                rpn_accuracy_for_epoch.append((len(pos_samples)))

                if C.num_rois > 1:
                    # If number of positive anchors is larger than 4//2 = 2, randomly choose 2 pos samples
                    if len(pos_samples) < C.num_rois//2:
                        selected_pos_samples = pos_samples.tolist()
                    else:
                        selected_pos_samples = np.random.choice(pos_samples,C.num_rois//2, replace=False).tolist()
                    
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

                    if C.train_verbose:
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
                        if C.train_verbose:
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
    '''
    
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


    
    