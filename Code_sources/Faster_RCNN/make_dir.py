import os
import glob
import shutil

IMAGES_FOLDER="permis/"

def readFileImages(strFolderName):
	print strFolderName
	image_list = []
	st=strFolderName+"*.png"
	for filename in glob.glob(st): #assuming gif
	    image_list.append(filename)
	return image_list

for i in range(0, 4):
	if not os.path.exists(str(i)):
		os.makedirs(str(i))
image_list=readFileImages(IMAGES_FOLDER)
j=0
for file in image_list:
	shutil.copy2(file, str(j%4)+"/")
	j=j+1



