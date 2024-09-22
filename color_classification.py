import cv2
from color_recognition_api import image_feature_extraction
from color_recognition_api import KNN
import os
import os.path
import sys




#Chay tap test va hien acurency
# KNN.main('./training_dataset/data.csv','./testing_dataset/testing.data')
# # checking whether the training data is ready
# PATH = './training_dataset/data.csv'

# """
# Ham isfile(): Kiem tra lieu mot duong dan cho truoc co ton tai hay khong
# Ham os.access(path, mode): kiem tra kha nang truy cap cua mot file nao do
#     + Return True or False
#     + Cac Mode: 
#         R_OK: read permission
#         F_OK: Existence
#         W_OK: Write permission

# """
# if os.path.isfile(PATH) and os.access(PATH, os.R_OK):
#     print ('training data is ready, classifier is loading...')
# else:
#     print ('training data is being created...')

def sol(test_img_path):
# read the test image
    try:
        source_image = cv2.imread(sys.argv[1])
    except:
        source_image = cv2.imread(test_img_path)
    prediction = 'n.a.'

    image_feature_extraction.color_histogram_of_test_image(source_image)
    prediction = KNN.classify_img('./testing_dataset/testing_image.data')

    img = cv2.resize(source_image, (450,400))
    img = cv2.copyMakeBorder(img, 10, 50, 10, 10, borderType=cv2.BORDER_CONSTANT, value=(50, 50, 50))
    cv2.putText(img, "Prediction: " + prediction, (40, 450), cv2.FONT_HERSHEY_DUPLEX, 1.3, (255, 255, 255))
    cv2.imshow('image', img)
    cv2.waitKey()
    # Display the resulting frame
    cv2.waitKey(0)		    