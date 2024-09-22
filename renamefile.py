import os
from os import path

cnt =1
old = ''
# print(os.listdir('./testing_image/Yellow'))
for f in os.listdir('./testing_image/Purple'):
    old_path = './testing_image/Purple/'+f
    if(f.endswith('.png')):
        tag= '.png'
    else:
        tag = '.jpg'
    print(f)
    print(old_path)

    os.rename(old_path,'./testing_image/Purple/' +'Purple' +str(cnt)+tag)
    cnt += 1