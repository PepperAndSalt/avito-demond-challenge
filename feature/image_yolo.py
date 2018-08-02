import darknet as dn
from tqdm import tqdm

import os
image_dir = r"..\Input\train_jpg_0"
arr = os.listdir(image_dir)
%%time
image_class = []
for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    image_class.append([f, classes])

image_dir = r"..\Input\train_jpg_1"
arr = os.listdir(image_dir)

for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    image_class.append([f, classes])

image_dir = r"..\train_jpg_2"
arr = os.listdir(image_dir)

for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    image_class.append([f, classes])

image_dir = r"..\Input\train_jpg_3"
arr = os.listdir(image_dir)

for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    image_class.append([f, classes])

image_dir = r"..\Input\train_jpg_4"
arr = os.listdir(image_dir)

for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    image_class.append([f, classes])

import pandas as pd
pd.DataFrame([[img, ','.join([c[0] for c in cls])] for img, cls in image_class], columns=['image','image_class']).to_csv('train_image_class.csv', index=False)

image_dir = r"..\Input\test_jpg\data\competition_files\test_jpg"
arr = os.listdir(image_dir)

test_image_class=[]
for f in tqdm(arr):
    full_image_path = os.path.join(image_dir, f)
#     print (f, full_image_path)
    classes = dn.performDetect(imagePath=full_image_path, 
              thresh= 0.25, 
              configPath = "cfg/yolov3.cfg", 
              weightPath = "yolov3.weights", 
              metaPath= "./data/coco.data", 
              showImage= False, 
              makeImageOnly = False, 
              initOnly= False)
    test_image_class.append([f, classes])
pd.DataFrame([[img, ','.join([c[0] for c in cls])] for img, cls in test_image_class], columns=['image','image_class']).to_csv('test_image_class.csv', index=False)

