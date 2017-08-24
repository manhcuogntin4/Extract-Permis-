import lxml.etree as et
import xml.etree.ElementTree as ET
import math
import cv2
import numpy as numpy
import imutils
import os
import glob
from shutil import copyfile


IMAGES_FOLDER="nom/"
DEST_DIR="nom1/"
def read_folder(fo):
    ls=[]
    for dirpath, dirnames, filenames in os.walk(fo):
        for fp in filenames:
            ls.append(fp)
    return ls

def readFileImages(strFolderName):
    print strFolderName
    image_list = []
    st=strFolderName+"*.png"
    for filename in glob.glob(st): #assuming gif
        image_list.append(filename)
    return image_list

def copy_file(file_input, file_output):
    copyfile(file_input, file_output)
    return file_output
def rotate_file_image(filename, Alpha):
    img=cv2.imread(filename,1)
    dst=imutils.rotate_bound(img,Alpha)
    w_n,h_n,d_n=dst.shape
    dirname= os.path.dirname(filename)
    filename=os.path.basename(filename)
    file_rotate_image=os.path.join(dirname, str(Alpha)+filename)
    Alpha=-1*Alpha
    cv2.imwrite(file_rotate_image,dst)

    file_annotate=os.path.join(dirname, filename).replace("png", "xml")
    file_annotate_rotate=file_rotate_image.replace("png", "xml")
    copy_file(file_annotate,file_annotate_rotate)
    rotate_xml(file_annotate_rotate, Alpha,w_n, h_n)

def rotate_file_clstm(filename, Alpha, dirname):
    img=cv2.imread(filename,1)
    dst=imutils.rotate_bound(img,Alpha)
    w_n,h_n,d_n=dst.shape
    filename=os.path.basename(filename)
    file_rotate_image=os.path.join(dirname, filename)
    Alpha=-1*Alpha
    cv2.imwrite(file_rotate_image,dst)

def rotate_xml(filename, Alpha, w,h):
    """ Parse a PASCAL VOC xml file """
    Alpha=math.radians(Alpha)
    tree = ET.parse(filename)
    objects = []


    for st in tree.findall('size'):
        w=int(st.find("width").text) -1
        h=int(st.find("height").text) -1
        print w, h
        print math.sin(Alpha)
        h_n=int(-w*math.sin(Alpha) + h*math.cos(Alpha))
        w_n=int(-h*math.sin(Alpha) + w*math.cos(Alpha))

        print w_n, h_n
        st.find("width").text=str(w_n)
        st.find("height").text=str(h_n)
        tree.write(filename)

    for obj in tree.findall('object'):
        # if not check(obj.find('name').text):
        #     continue
        obj_struct = {}
        obj_struct['name'] = obj.find('name').text
        bbox = obj.find('bndbox')
        obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                              int(bbox.find('ymin').text) - 1,
                              int(bbox.find('xmax').text) - 1,
                              int(bbox.find('ymax').text) - 1]
        # Read the original coordination
        x1=int(bbox.find('xmin').text)
        y1=int(bbox.find('ymin').text)
        x2=int(bbox.find('xmax').text)
        y2=int(bbox.find('ymax').text)
        # Transformation
        y1_t=int(y1*math.cos(Alpha) - x1*math.sin(Alpha))
        x1_t=int(x1*math.cos(Alpha) - y1*math.sin(Alpha)) 
        x2_t=int(x2*math.cos(Alpha) - y2*math.sin(Alpha))
        y2_t=int(y2*math.cos(Alpha) - x2*math.sin(Alpha))

        # Transformation


        #x1_n=int((x1*math.cos(Alpha)-y1*math.sin(Alpha)) - (w_n-y2*(1-math.tan(Alpha)))*math.sin(Alpha))
        #x2_n=x1_n+int((x2-x1)*math.cos(Alpha)+(y2-y1)*math.sin(Alpha))
        x1_n=x1_t - int((h_n-y1)*math.sin(Alpha)) + int(y2*math.sin(Alpha))
        x2_n=x1_n+ int((x2-x1)*math.cos(Alpha) -(y2-y1)*math.sin(Alpha))
        

        y1_n=y1_t #- int((w_n-x2)*math.sin(Alpha)) - int(x1*math.tan(Alpha))
        y2_n=y2_t #- int((w_n-x2)*math.sin(Alpha)) - int(x1*math.tan(Alpha))
        if(y2_n>h_n):
             y2_n=h_n
        if(x2_n>w_n):
             x2_n=w_n
        if x2_n <= x1_n or y2_n<=y1_n:
            print filename
        bbox.find('xmin').text=str(x1_n)
        bbox.find('ymin').text=str(y1_n)
        bbox.find('xmax').text=str(x2_n)
        bbox.find('ymax').text=str(y2_n)
        tree.write(filename)


#file_image="test1.png"
#rotate_file_image(file_image, 3)
if not os.path.exists(DEST_DIR):
                            os.makedirs(DEST_DIR)

for filepath in readFileImages(IMAGES_FOLDER):
        rotate_file_clstm(filepath,2, DEST_DIR)