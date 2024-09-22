
from PIL import Image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


def color_histogram_of_test_image(test_src_image):

    # load the image
    image = test_src_image

    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        print('chanel '+ str(counter))
        print(hist.shape)        

        # print(hist)
        # find the peak pixel values for R, G, and B
        elem = np.argmax(hist)
        print(elem)
        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue
            # print(feature_data)

    # Mo file va ghi de len file data
    with open('./testing_dataset/testing_image.data', 'w') as myfile:
        myfile.write(feature_data)


def color_histogram_of_image(img_name):

    # detect image color by using image file name to label training data
    
    label =''
    for file_name in os.listdir('./testing_image'):
        if(file_name in img_name):
            label = file_name
            break


    # load the image
    """
    Doc file anh va tra ve mot ma tran 3 chieu ung voi cac kenh mau B, G, R
    """
    image = cv2.imread(img_name)

    """
    _Tach anh mau thanh cac kenh mau B, G, R 
    _ Tu mot ma tran 3 chieu --> 3 ma tran 2 chieu tuong ung voi cac kenh mau B, G, R
    _ Khi tach anh theo tung kenh mau, buc anh se chuyen ve dang grayscale, co the xem trong pts de hieu hon
    """
    chans = cv2.split(image)
    colors = ('b', 'g', 'r')
    features = []
    feature_data = ''
    counter = 0

    """
        _ Ham zip se ghep cac phan tu cua 2 iterable khac nhau thanh cap cac tupple
        _ Neu 2 iterable co do dai khac nhau
        _ Vong lap for thuc hien viec gan 3 kenh mau(R,G,B) voi nhan cua no
    """
    for (chan, color) in zip(chans, colors):
        counter = counter + 1

        """
        calcHist: tra ve tan suat xuat hien cua cac gia tri cuong do mau trong kenh mau do
        Ma tran co kich thuoc la 256x1
        _ Moi dong se dai dien cho mot cuong do mau cu the
        _ Gia tri tai moi dong la tan xuat(so luong pixel) co muc do mau tuong ung
        _ Vi du neu gia tri tai hang 125 la 50 --> co 50px trong buc anh chua cuong do mau 125
        """
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        features.extend(hist)

        # find the peak pixel values for R, G, and B
        """
        _ Tim ra muc xam co so luong diem anh cao nhat trong cac kenh mau, lay no lam dai dien cho kenh mau
        _ argmax: Tim ra chi so co gia tri lon nhat trong mot mang 
        """
        elem = np.argmax(hist)

        """
        Gan muc xam cac bien
        """
        if counter == 1:
            blue = str(elem)
        elif counter == 2:
            green = str(elem)
        elif counter == 3:
            red = str(elem)
            feature_data = red + ',' + green + ',' + blue

    with open('./testing_dataset/testing.data', 'a') as myfile:
        myfile.write(feature_data + ',' + label + '\n')


def training():

    # black color training images
    for f in os.listdir('./testing_image/Black'):
        color_histogram_of_image('./testing_image/Black/' + f)

    # blue color training images
    for f in os.listdir('./testing_image/Blue'):
        color_histogram_of_image('./testing_image/Blue/' + f)		

    # brown color training images
    for f in os.listdir('./testing_image/Brown'):
        color_histogram_of_image('./testing_image/Brown/' + f)		

    # green color training images
    for f in os.listdir('./testing_image/Green'):
        color_histogram_of_image('./testing_image/Green/' + f)

    # grey color training images
    for f in os.listdir('./testing_image/Grey'):
        color_histogram_of_image('./testing_image/Grey/' + f)	
    
    # orange color training images
    for f in os.listdir('./testing_image/Orange'):
        color_histogram_of_image('./testing_image/Orange/' + f)


    # red color training images
    for f in os.listdir('./testing_image/Red'):
        color_histogram_of_image('./testing_image/Red/' + f)

    # purple color training images
    for f in os.listdir('./testing_image/Purple'):
        color_histogram_of_image('./testing_image/Purple/' + f)	

    # white color training images
    for f in os.listdir('./testing_image/White'):
        color_histogram_of_image('./testing_image/White/' + f)

    # yellow color training images
    for f in os.listdir('./testing_image/yellow'):
       color_histogram_of_image('./testing_image/Yellow/' + f)


image = cv2.imread('./testing_image/test_img3.png')
color_histogram_of_test_image(image)
# training()