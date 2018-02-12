# coding=utf-8
import glob
import sys

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-f", "--folder", required=True,
    help="images folder")
args = vars(ap.parse_args())

ANNOT_FOLDER=args["folder"]
MRZ_LEN = 36
def readFileFolder(strFolderName):
	print strFolderName
	file_list = []
	st=strFolderName+"*.txt"
	for filename in glob.glob(st): #assuming gif
	    file_list.append(filename)
	return file_list

def remove_empty_line(filename):
	with open(filename) as xmlfile:
	    lines = [line for line in xmlfile if line.strip() is not ""]

	with open(filename, "w") as xmlfile:
	    xmlfile.writelines(lines)

def correct_file_text(filename):
	lines=[]
	with open(filename, 'r') as file: 
		for line in file:
			if len(line.strip())<MRZ_LEN and line.strip() is not "":
				print filename
				print len(line)
				line = line +''.join(['<' for i in range(MRZ_LEN - len(line.strip()))])
			lines.append(line)
				#break
	with open(filename, 'w') as text_file:
	    l=""
	    for line in lines:
	    	l+=line.rstrip('\n')
	    l=l.replace("\n", "")
	    if len(l) !=0:
	    	text_file.write(l)	

ls_files=readFileFolder(ANNOT_FOLDER)


for filename in ls_files:
	correct_file_text(filename)
	#remove_empty_line(filename)
