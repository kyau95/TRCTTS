import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from PIL import ImagePath
import pandas as pd
import os
from os import path
import pathlib
import re
import pickle

# Get path to images
dir_path = "./images/"
data_root = pathlib.Path(dir_path)
print(data_root)

# Creating dataframe for all image paths
def return_file_names_df(root_dir, x):
    arr = os.listdir(root_dir)
    col = []
    for dir in arr:
        if dir == x:
            dir_path = str(root_dir) + '/' + dir
            data_root= pathlib.Path(dir_path)
            li= list(sorted(data_root.glob('*.*')))
            li = [str(path) for path in li]
            col.append(li)
    data_df = pd.DataFrame()
    data_df['path_img'] = col[0]
    return data_df

df_train = return_file_names_df(data_root, "train")
df_test = return_file_names_df(data_root, "test")

""">>> Checking dimensions of images in training and testing images <<<"""
# print(df_train.head())
# dim, channel, extnsn = [], [], []
# for path in df_train['path_img'].values:
#     img = cv2.imread(path)
#     dim.append(img.shape[:2])
#     channel.append(img.shape[2])
#     extnsn.append(path.split('.')[-1])
# print('Dimension of all images:', set(dim))
# print('No. channels of all images:', set(channel))
# print('Extesions of all images:', set(extnsn))

# # print(df_test.head())
# dim, channel, extnsn = [], [], []
# for path in df_test['path_img'].values:
#     img = cv2.imread(path)
#     dim.append(img.shape[:2])
#     channel.append(img.shape[2])
#     extnsn.append(path.split('.')[-1])
# print('Dimension of all images:', set(dim))
# print('No. channels of all images:', set(channel))
# print('Extesions of all images:', set(extnsn))

# Getting sizes of images
heights = []
widths = []
for image in df_train['path_img']:
    img = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    shape = img.shape
    heights.append(shape[0])
    widths.append(shape[1])

# Manually coordinates and text of the imgaes to df
def text_coord(df):
    coordinates = []
    texts = []
    for path in df['path_gt']:
        li = []
        f = open(str(path), "r",encoding='utf-8')
        for x in f:
            li.append(x.split(','))
            ji = []
        for i in li:
            a = i[-1]
            a = re.sub(r"[\n\t\-\\\/]","",a)
            a = a.lower()
            ji.append(a)
            az = li
        for i in az:
            i.remove(i[-1])
        coordinates.append(az)
    texts.append(ji)
    return texts, coordinates

texts, coordinates = text_coord(df_train)
df_train['texts'], df_train['coordinates'] = texts, coordinates
# print(df_train.head())

#for test
texts_test, coordinates_test = text_coord(df_test)
df_test['texts'],df_test['coordinates'] = texts_test,coordinates_test
# print(df_test.head())

# Drawing bounding boxes for given image
def bounding_box(img_array,df_txt,df_co):
  txt=np.array(df_txt)
  v = []
  for i in df_co:
    v.append(list(map(int,i)))

  b=np.array(v)
  (r, c) = b.shape
  for y in range(0, r):
    for x in range(0, c):
      rec_pts = np.array([[b[y,0],b[y,1]],[b[y,2],b[y,3]],[b[y,4],b[y,5]],[b[y,6],b[y,7]]], np.int32) # taking the coordinates of rectangle
      #rec_pts = pts.reshape((-1,1,2))
      img = cv2.polylines(img_array,[rec_pts],True,(0,255,255),thickness =3) # used to draw bounding box on the image
      
      (text_width, text_height) = cv2.getTextSize(txt[y], cv2.FONT_HERSHEY_PLAIN, 1.5, 1)[0] ## taking width and height of the image
      # set the text start position
      text_offset_x = b[y,0] 
      text_offset_y = b[y,1] + 2
      # make the coords of the box with a small padding of two pixels
      box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
      cv2.rectangle(img, box_coords[0], box_coords[1], (255, 255, 255), cv2.FILLED)
      cv2.putText(img, txt[y], (text_offset_x, text_offset_y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 1) # put the text on the image
  plt.grid(False)      
  plt.imshow(img)

# Testing method on Training data
for i in range(90,95):
  filepath = df_train['path_img'][i]
  txt_p = df_train['texts'][i]
  co_d = df_train['coordinates'][i]

  img_array = cv2.imread(filepath)

  plt.figure(figsize=(18,8))
  plt.subplot(1,2,1)
  plt.grid(False)   
  plt.imshow(img_array)

  plt.subplot(1,2,2)
  bounding_box(img_array,txt_p,co_d)
  plt.savefig('plot'+str(i)+'.png', dpi=300, bbox_inches='tight')

# Testing method on Test data
for i in range(100, 105):
  filepath = df_test['path_img'][i]
  txt_p = df_test['texts'][i]
  co_d = df_test['coordinates'][i]

  img_array = cv2.imread(filepath)

  plt.figure(figsize=(18,8))
  plt.subplot(1,2,1)
  plt.grid(False)   
  plt.imshow(img_array)

  plt.subplot(1,2,2)
  bounding_box(img_array,txt_p,co_d)
  plt.savefig('plot'+str(i)+'.png', dpi=300, bbox_inches='tight')

with open("./df_train.pkl", "w") as train_file:
    pickle.dump(train_file, df_train)

with open("./df_test.pkl", "w") as test_file:
    pickle.dump(test_file, df_test)