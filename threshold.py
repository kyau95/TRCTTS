"""
Testing MSER and SWT pre-processing
"""

import cv2
import pandas as pd
import matplotlib.pyplot as plt


df_train = pd.read_pickle("df_train.pkl")
df_test = pd.read_pickle("df_test.pkl")

#Create MSER object

for i in range(10,15):
    filepath = df_train['path_img'][i]
    img_array = cv2.imread(filepath)

    plt.figure(figsize=(18,8))
    plt.subplot(1,2,1)
    plt.grid(False)   
    plt.imshow(img_array)
    plt.subplot(1,2,2)
    mser = cv2.MSER_create()

    #Reading image
    img = cv2.imread(df_train['path_img'].values[i])
    #Convert to gray scale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #Detect regions in gray scale image
    regions, _ = mser.detectRegions(gray)

    #Hulls for each dtected regions
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    #Drawing polylines on image
    cv2.polylines(img, hulls, 1, (0, 255, 0))

    #Showing image with polylines on detected text in an image
    plt.grid(False)
    plt.imshow(img)

#https://stackoverflow.com/questions/11116199/stroke-width-transform-swt-implementation-python
from swtloc import SWTLocalizer
from swtloc.utils import imgshowN, imgshow

swtl = SWTLocalizer()
imgpaths = ... # Image paths, can be one image or more than one
swtl.swttransform(imgpaths=df_train['path_img'][13], save_results=True, save_rootpath='swtres/',
                  edge_func = 'ac', ac_sigma = 1.0, text_mode = 'db_lf',
                  gs_blurr=True, blurr_kernel = (5,5), minrsw = 3, 
                  maxCC_comppx = 10000, maxrsw = 200, max_angledev = np.pi/6, 
                  acceptCC_aspectratio = 5.0)

imgshowN([swtl.orig_img, swtl.swt_mat], ['Original Image', 'Stroke Width Transform'])

respacket = swtl.get_grouped(lookup_radii_multiplier=1, sw_ratio=2,
                             cl_deviat=[13,13,13], ht_ratio=2, 
                             ar_ratio=3, ang_deviat=30)

grouped_labels = respacket[0]
grouped_bubblebbox = respacket[1]
grouped_annot_bubble = respacket[2]
grouped_annot = respacket[3]
maskviz = respacket[4]
maskcomb  = respacket[5]

imgshowN([swtl.orig_img, swtl.swt_labelled3C, grouped_annot_bubble],
         ['Original', 'SWT','Bubble BBox Grouping'],figsize=(15,8))