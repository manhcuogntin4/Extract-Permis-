import requests
import glob
import sys
import json
import io
import os.path


def readFileFolder(strFolderName, suffix):
	print strFolderName
	file_list = []
	st=strFolderName+"*."+suffix
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list

def readBaseNameFolder(strFolderName, suffix):
	print strFolderName
	file_list = []
	st=strFolderName+"*."+suffix
	for filename in glob.glob(st): #assuming gif
	    basename=os.path.splitext(os.path.basename(filename))[0]
	    file_list.append(basename)
	return file_list

def removeFileText(ls_imageFile, ls_txtFile):
	for filetext in ls_txtFile:
		if filetext not in ls_imageFile:
			file_name=filetext+".txt"
			os.remove(file_name)


ls_imageFiles=readBaseNameFolder("./", "png")
print ls_imageFiles
ls_txtFiles=readBaseNameFolder("./","txt")
print ls_txtFiles

removeFileText(ls_imageFiles, ls_txtFiles)