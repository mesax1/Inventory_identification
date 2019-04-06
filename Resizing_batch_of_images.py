# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 14:36:13 2019

@author: JUAN
"""

# -*- coding: utf-8 -*-

''' Script to resize all files in current directory,
    saving new .jpg and .jpeg images in a new folder. '''


import cv2
import glob
import os
import shutil


rootdir = 'C:/Users/JUAN/Documents/Deep_Learning_A_Z/Volume 1 - Supervised Deep Learning/Part 2 - Convolutional Neural Networks (CNN)/FOTOS LABORATORIO - copia'

filenames= os.listdir (".") # get all files' and folders' names in the current directory


for filename in filenames:
    try: 
        os.chdir(rootdir+'/'+filename) 
    except: 
        print('not a folder')
    folder = 'resized'+filename
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # Get images
    imgs = glob.glob('*.jpg')
    imgs.extend(glob.glob('*.jpeg'))

    print('Found files:')
    print(imgs)

    width = 128
    print('Resizing all images be %d pixels wide' % width)


    # Iterate through resizing and saving
    for img in imgs:
        pic = cv2.imread(img, cv2.IMREAD_UNCHANGED)
        (h, w) = pic.shape[:2]
        #Crop images before resizing, in order to obtain squared images
        if h>w:
            base=int((h-w)/2)
            cropped = pic[base:w+base, 0:w]
        else :
            base=int((w-h)/2)
            cropped = pic[0:h, base:h+base]
        height = int(width * cropped.shape[0] / cropped.shape[1])
        pic = cv2.resize(cropped, (width, height))
        cv2.imwrite(folder + '/' + img, pic)

    root_src_dir = folder
    #Create two target directories, one for training_set and one for test_set
    #root_target_dir1 = rootdir+'/'+'resized'+'/'+'training_set'+'/'+filename 
    root_target_dir1 = rootdir+'/'+'resized'+'/'+'dataset'+'/'+filename 
    root_target_dir2 = rootdir+'/'+'resized'+'/'+'test_set'+'/'+filename

    operation= 'copy' # 'copy' or 'move'
    cont = 0
    for src_dir, dirs, files in os.walk(root_src_dir):
        for file_ in files:
            if cont<5:
                dst_dir = src_dir.replace(root_src_dir, root_target_dir1)
                #cont=cont+1
            #if cont>=5: 
              #  dst_dir = src_dir.replace(root_src_dir, root_target_dir2)
               # cont=0
                    # dst_dir = filename
            if not os.path.exists(dst_dir):
                os.mkdir(dst_dir)
            src_file = os.path.join(src_dir, file_)
            dst_file = os.path.join(dst_dir, file_)
            if os.path.exists(dst_file):
                os.remove(dst_file)
            if operation is 'copy':
                shutil.copy(src_file, dst_dir)
            elif operation is 'move':
                shutil.move(src_file, dst_dir)
    os.chdir(rootdir)
