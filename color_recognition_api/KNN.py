import math
import csv
import random
import operator
import cv2
import os
import pandas as pd
import numpy as np

def calculateDistance(var1, var2):
    length= 3 # Do mot mau co 3 he mau (RGB)
    distance = 0
    for i in range(length):
        distance += pow((var1[i] - var2[i]), 2)
    return math.sqrt(distance)


def k_NearestNeighbors(training_data, test_instance, k):
    list_neighbor =[] # list chua cac tuple voi gia tri dau la label, gia tri 2 la khoang cach
    for x in training_data: 
        distance = calculateDistance(x, test_instance) # Lan luot tinh khoang cach giua du lieu 
        list_neighbor.append((x[-1], distance))
    list_neighbor.sort(key=lambda x: x[1])
    return list_neighbor[:k] # Tra ve k tuple dau tien


def findMostOccur(list_neighbor):
    frequency = {} # 1 dict luu tan suat xuat hien
    for neighbor in list_neighbor:
        if neighbor[0] in frequency:
            frequency[neighbor[0]] +=1
        else:
            frequency[neighbor[0]] = 1
    
    res = sorted(frequency.items(), key= lambda x: x[1], reverse= True) # Sap xep theo tan suat xuat hien
    return res[0][0] # Tra ve gia tri dau tien cua dict (label)


def loadData(path):
    """
    Ham isfile(): Kiem tra lieu mot duong dan cho truoc co ton tai hay khong
    Ham os.access(path, mode): kiem tra kha nang truy cap cua mot file nao do
        + Return True or False
        + Cac Mode: 
            R_OK: read permission
            F_OK: Existence
            W_OK: Write permission
    """
    if os.path.isfile(path) and os.access(path, os.R_OK):
        with open(path) as csvfile:
            lines = csv.reader(csvfile)
            data = list(lines) # Chuyen file csv ve dang list 2 chieu

            if 'training' in path:
                data.pop(0) # Xoa header neu la tap training
                
            for x in range(len(data)):
                for y in range(3):
                    data[x][y] = float(data[x][y])
            return data

# loadData('./training_dataset/data.csv')


def main(training_path, testing_path):# Du daon file testting.data va hien predicttions
    training_vector=[]
    testing_vector = []

    training_vector = loadData(training_path)
    testing_vector = loadData(testing_path)
    k = 5

    cnt =0
    prediction=[]
    for i in range(len(testing_vector)):
        list_neighbor = k_NearestNeighbors(training_vector, testing_vector[i], k)
        result = findMostOccur(list_neighbor)
        prediction.append(result)
        if result == testing_vector[i][-1]:
            cnt +=1
            print(str(i+1)+ ' '+ str(testing_vector[i]) + '-->'+result)    
        else:
            print(str(i+1)+ ' '+ str(testing_vector[i]) + '-->'+result + '(Wrong)')             
    # print(cnt)
    print('Acurency: '+ str(cnt/len(testing_vector) * 100) )
    # print(prediction) 
    # print(testing_vector)
    # print(type(training_vector))

def classify_img(testing_path):
    training_vector=[]
    testing_vector = []

    training_vector = loadData('./training_dataset/data.csv')
    testing_vector = loadData(testing_path)
    # print(testing_vector)
    k = 5
    list_neighbor = k_NearestNeighbors(training_vector, testing_vector[0], k)
    result = findMostOccur(list_neighbor)
    # print(result)
    return result

# main('./training_dataset/data.csv', './testing_dataset/testing.data')
# classify_img('./testing_dataset/testing_image.data')