from collections import defaultdict
from scipy.stats import itemfreq
from scipy import ndimage as ndi
import matplotlib.pyplot as plt
from skimage import feature
from PIL import Image as IMG
import numpy as np
import pandas as pd 
import operator
import cv2
import os 

from IPython.core.display import HTML 
from IPython.display import Image
from zipfile import ZipFile

features = pd.DataFrame()
zipped = ZipFile('test_jpg.zip')
# filenames = zipped.namelist()[1:] # exclude the initial directory listing for train_jpg
filenames = zipped.namelist()[1:] # do not need to exclude for small train_jpg
features['image'] = filenames

####### Feature 1 : Dullness ######
def color_analysis(img):
    # obtain the color palatte of the image 
    palatte = defaultdict(int)
    for pixel in img.getdata():
        palatte[pixel] += 1
    
    # sort the colors present in the image 
    sorted_x = sorted(palatte.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness 
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness 
            light_shade += x[1]
        shade_count += x[1]
        
    light_percent = round((float(light_shade)/shade_count)*100, 2)
    dark_percent = round((float(dark_shade)/shade_count)*100, 2)
    return light_percent, dark_percent

def perform_color_analysis(img, flag):
    try:
        pre_open = zipped.open(img)         
        im = IMG.open(pre_open)  
#     path = images_path + img 
#     im = IMG.open(path) #.convert("RGB")
    
    # cut the images into two halves as complete average may give bias results
        size = im.size
        halves = (size[0]/2, size[1]/2)
        im1 = im.crop((0, 0, size[0], halves[1]))
        im2 = im.crop((0, halves[1], size[0], size[1]))

        try:
            light_percent1, dark_percent1 = color_analysis(im1)
            light_percent2, dark_percent2 = color_analysis(im2)
        except Exception as e:
            return None

        light_percent = (light_percent1 + light_percent2)/2 
        dark_percent = (dark_percent1 + dark_percent2)/2 
    
        if flag == 'black':
            return dark_percent
        elif flag == 'white':
            return light_percent
        else:
            return None
    except:
        pass

features['dullness']= features['image'].apply(lambda x : perform_color_analysis(x, 'black'))


###### Feature 2: Image Whiteness ######
features['whiteness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'white'))

###### Feature 3: Average Pixel Width ######
def average_pixel_width(img):
    try:        
        pre_open = zipped.open(img)         
        im = IMG.open(pre_open)     
        im_array = np.asarray(im.convert(mode='L'))
        edges_sigma1 = feature.canny(im_array, sigma=3)
        apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
        f_apw = apw*100
        return f_apw
    except:
        pass

features['average_pixel_width'] = features['image'].apply(lambda x : average_pixel_width(x))


###### Feature 4: Average Color ######
def get_average_color(img):
    try:
        pre_open = zipped.read(img)         
        img = cv2.imdecode(np.frombuffer(pre_open, np.uint8), 1)
        average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
        return average_color
    except:
        pass

features['average_color'] = features['image'].apply(get_average_color)

def get_red(oda):
    try:
        return oda[0] / 255
    except:
        pass
    
def get_green(oda):
    try:
        return oda[1] / 255
    except:
        pass
    
def get_blue(oda):
    try:
        return oda[2] / 255
    except:
        pass
features['average_red'] = features['average_color'].apply(lambda x: get_red(x)) 
features['average_green'] = features['average_color'].apply(lambda x: get_green(x)) 
features['average_blue'] = features['average_color'].apply(lambda x: get_blue(x)) 
features.drop('average_color', axis=1, inplace=True)


###### Feature 5 Dimensions of the Image ######

# create a temp index1 column
features['index1'] = features.index

filelist = zipped.filelist[0:]
def getSize(fileindex):
    try:
        img_size = filelist[fileindex].file_size
        return img_size
    except:
        pass

def getDimensions(filename):
    try:
        pre_open = zipped.open(filename)         
        img_size = IMG.open(pre_open).size
        return img_size 
    except:
        pass

features['image_size'] = features['index1'].apply(getSize)
features['temp_size'] = features['image'].apply(getDimensions)
def get_width(oda):
    try:
        return oda[0]
    except:
        pass
    
def get_height(oda):
    try:
        return oda[1]
    except:
        pass

features['width'] = features['temp_size'].apply(lambda x : get_width(x))
# features = features.drop(['temp_size', 'average_color', 'dominant_color'], axis=1)
features = features.drop(['temp_size'], axis=1)
features.drop(['index1'], axis=1, inplace=True)
features['height'] = features['temp_size'].apply(lambda x : get_height(x))


###### Feature 6 Image Blurrness ######
def get_blurrness_score(image):
    try:
        pre_open = zipped.read(image)         
        image = cv2.imdecode(np.frombuffer(pre_open, np.uint8), 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = cv2.Laplacian(image, cv2.CV_64F).var()
        return fm
    except:
        pass

features['blurrness'] = features['image'].apply(get_blurrness_score)
features.to_csv('test_image_feature.csv', index=False)