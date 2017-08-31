import cv2
import numpy as np
from matplotlib import pyplot as plt
import pyclstm
from PIL import Image
import sys, getopt
import os
import difflib
import sys

#Process parallel
from multiprocessing import Pool   
import multiprocessing 
import subprocess
from multiprocessing import Manager
from functools import partial
import multiprocessing.pool
import itertools



reload(sys)
sys.setdefaultencoding('utf-8')
CACHE_FOLDER = '/tmp/caffe_demos_uploads/cache'
this_dir = os.path.dirname(__file__)

def get_similar(str_verify, cls="ville", score=0.7):
	words = []
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	if(isLieu):
		lieu_path=os.path.join(this_dir, 'lieu.txt')	
	else:
		lieu_path=os.path.join(this_dir, 'nom.txt')			
	f=open(lieu_path,'r')
	for line in f:
		words.append(line)
	f.close()
	simi=difflib.get_close_matches(str_verify, words,5,score)
	return simi

def convert_to_binary(img):
	if (img.shape >= 3):
		img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	ret, imgBinary = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	height = np.size(img, 0)
	width = np.size(img, 1)
	height=60
	r,c=img.shape[:2]
	print (height*c)/r, imgBinary.size
	res = cv2.resize(imgBinary,((int)(height*c)/r, height), interpolation = cv2.INTER_CUBIC)
	res = cv2.fastNlMeansDenoising(res,20, 7, 21)
	out_path = os.path.join(CACHE_FOLDER, str(os.getpid())+ "out.png")
	cv2.imwrite(out_path,res)
	return out_path, res


def extract_text(img_path, model_path):
	print "extract text"
	ocr = pyclstm.ClstmOcr()
	ocr.load(model_path)
	imgFile = Image.open(img_path)
	#print "file image opened"
	text=""
	try:
		text = ocr.recognize(imgFile)
		text.encode('utf-8')
	except: 
		print "error string"
	
	#print "ocr success"
	chars=[]
	try:
		chars = ocr.recognize_chars(imgFile)
	except:
		print "error chars"
	print "pass"
	prob = 1
	prob1=1
	index = 0
	if(text.strip().find(u' ') != -1 and (text.strip().index(u' ') <= 3)):
		if(len(text)>text.index(u' ')+1):		
			index = text.index(u' ')+1
	for ind, j in enumerate(chars):
		if ind >= 0:		
			prob *= j.confidence
		if ind >=index:
			prob1 *=j.confidence
	print os.getpid()
	if prob<0.7:
		return text[index:], prob1,0
	return text[0:], prob, 0


def crop_image(img,cropX=0, cropY=0, cropWidth=0, cropHeight=0):
	h = np.size(img, 0)
	w = np.size(img, 1)	
	if h<=cropY or w<=cropX:
		print "Error size image" 
	if(h-cropHeight>cropY) and (w-cropWidth>cropX):
		res=img[cropY:h-cropHeight, cropX:w-cropWidth]
	else:
		res=img[cropY:h,cropX:w]
	#print str(os.getpid())
	out_path = os.path.join(CACHE_FOLDER, str(os.getpid())+"croped.png")
	cv2.imwrite(out_path,res)
	return out_path


def clstm_ocr_permis(img, cls="nom"):
	# if not os.path.exists(CACHE_FOLDER):
	# 	os.makedirs(CACHE_FOLDER)

	try:
		os.mkdir(CACHE_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(CACHE_FOLDER, 0755)

	if cls=="nom" or cls=="prenom":
		print "nom prenom"
		model_path = os.path.join(this_dir, 'model_nomprenom_permis_final.clstm')
	
	else:
		print "date"
		model_path = os.path.join(this_dir, 'model_date_permis.clstm')
	if(img.size>0):
		converted_image_path, image = convert_to_binary(img)
		#maxPro = 0
		#ocr_result = ""
		ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
		#print "extract text success"
	else:
		return ("",0)

	if(index>0):
		image=image[10:,:]
	cropX=4
	cropY=6
	cropWidth=1
	cropHeight=4
	if cls=="date_naissance" or cls=="date_permis_B1":
		cropHeight=6
		cropWidth=2
	if cls=="nom":
		cropX=6
		cropY=4
	if cls=="prenom":
		cropX=6
		cropHeight=6
		#cropWidth=2


	if cls in ['date_naissance', 'date_permis_A1', \
                         'date_permis_A2', 'date_permis_A3', 'date_permis_B1', 'date_permis_B']:
		if len(ocr_result)!=10:
			#print "date_naissance"
			maxPro=0	

	for i in range (0,cropX,1):
		for j in range (0,cropY):
			for k in range (0,cropWidth):
				for h in range (0, cropHeight):
					img_path = crop_image(image, 4*i, 3*j, 4*k, 3*h)
					text, prob, index = extract_text(img_path, model_path)
					if(prob > maxPro) and (len(text)>=2) and checkdate(text, cls):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2):
					 	break	
					# else:
					# 	if (maxPro > 0.97):
					# 		break
	return (ocr_result, maxPro)

def checkdate(text, cls="nom"):
	if cls=="nom" or cls=="prenom" or ( len(text)==10):
		return True
	else:
		return False

def clstm_ocr_permis_parallel(img, cls="nom"):
	# if not os.path.exists(CACHE_FOLDER):
	# 	os.makedirs(CACHE_FOLDER)

	try:
		os.mkdir(CACHE_FOLDER)
	except OSError as e:
		if e.errno == 17:  # errno.EEXIS
			os.chmod(CACHE_FOLDER, 0755)

	if cls=="nom" or cls=="prenom":
		print "nom prenom"
		#model_path = os.path.join(this_dir, 'model_nom_carte_grise_090317.clstm')
		model_path = os.path.join(this_dir, 'model_nomprenom_permis_final.clstm')
		#model_path = os.path.join(this_dir, 'model_nom_carte_grise_120317x3.clstm')
	else:
		print "date"
		model_path = os.path.join(this_dir, 'model_date_permis.clstm')
	
	if(img.size>0):
		converted_image_path, image = convert_to_binary(img)
		#maxPro = 0
		#ocr_result = ""
		ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
	else:
		return ("",0)

	print "ocr_permis_parallel"
	if(index>0):
		image=image[10:,:]
	cropX=4
	cropY=6
	cropWidth=1
	cropHeight=4
	if cls=="date_naissance" or cls=="date_permis_B1":
		cropHeight=6
		#cropWidth=2
	if cls=="nom":
		cropX=6
		cropY=4
	if cls=="prenom":
		cropX=6
		cropHeight=6
	q={}
	p={}
	txt={}
	prob={}
	for j in range (0,cropY):
		q[j] = multiprocessing.Queue()
		p[j] = multiprocessing.Process(target=calib_clstm_height_low_queue, args=(cropX,j,cropWidth,cropHeight,image,model_path,ocr_result, maxPro, cls, q[j]))
		p[j].start()
		#xt[j],prob[j]=q[j].get()
		# if(q[j].empty()==False):
		# 	txt[j],prob[j]=q[j].get()
		# else:
		# 	txt[j],prob[j]="",0

	for j in range (0,cropY):
		txt[j],prob[j]=q[j].get()
	
	for j in range (0,cropY):
		p[j].join()
	
		
	for j in range (0,cropY):
		if(prob[j]>maxPro):
			maxPro=prob[j]
			ocr_result=txt[j]

	# Pool 
	# Y=[0,1,2,3,4,5,6]
	# p=Pool(7)
	# ar=[(cropX,0,cropWidth,cropHeight,image,model_path), (cropX,1,cropWidth,cropHeight,image,model_path), (cropX,2,cropWidth,cropHeight,image,model_path), 
	# (cropX,3,cropWidth,cropHeight,image,model_path), (cropX,4,cropWidth,cropHeight,image,model_path), (cropX,5,cropWidth,cropHeight,image,model_path), (cropX,6,cropWidth,cropHeight,image,model_path)]
	# res =p.map(calib_clstm_height_low_new,a)
	# p.close()
	# p.join()
	# ocr_result, maxPro="res",1
	#for j in Y:
	#	print j
		# txt[j],prob[j]=res[j]
		# if(prob[j]>maxPro):
		# 	maxPro=prob[j]
		# 	ocrresult=txt[j]

	return (ocr_result, maxPro)






def clstm_ocr_calib_permis(img, cls="nom"):
	if not os.path.exists(CACHE_FOLDER):
		os.makedirs(CACHE_FOLDER)
	if cls=="nom" or cls=="prenom":
		print "clstm_ocr_calib_permis"
		#model_path = os.path.join(this_dir, 'model_nom_carte_grise_090317.clstm')
		model_path = os.path.join(this_dir, 'model_nomprenom_permis_final.clstm')
		#model_path = os.path.join(this_dir, 'model_nom_carte_grise_120317x3.clstm')
	else:
		print "date"
		model_path = os.path.join(this_dir, 'model_date_permis.clstm')
	ocr_result, maxPro="",0
	if(img.size>0):
		print "Covert " 
		converted_image_path, image = convert_to_binary(img)
		print "Covert pass"
	#maxPro = 0
	#ocr_result = ""
		ocr_result, maxPro, index=extract_text(converted_image_path, model_path)
		print "calib result", ocr_result, maxPro
	else:
		return ("",0)
	return (ocr_result, maxPro)

def calib_clstm_height_low(cropX, y, cropWidth, cropHeight, image, model_path, ocr_result_n, maxPro_n, cls):
	maxPro=maxPro_n
	ocr_result=ocr_result_n
	for i in range (0,cropX,1):
			for k in range (0,cropWidth):
				for h in range (0, cropHeight):
					img_path = crop_image(image, 4*i, 3*y, 4*k, 3*h)
					text, prob, index = extract_text(img_path, model_path)
					os.remove(img_path)
					if(prob > maxPro) and (len(text)>=2) and checkdate(text, cls):
						maxPro = prob
						ocr_result = text
					if (maxPro > 0.95) and (len(text) >= 2) and checkdate(text, cls):
						break
	return ocr_result, maxPro




def calib_clstm_height_low_queue(cropX, y, cropWidth, cropHeight, image, model_path, ocr_result_n, maxPro_n, cls, q):
	q.put(calib_clstm_height_low(cropX, y, cropWidth, cropHeight, image, model_path, ocr_result_n, cls, maxPro_n))

if __name__ == '__main__':
	#filename = os.path.join(this_dir, 'demo', 'prenom0.png')
	filename="nom2.png"
	img = cv2.imread(filename,1)
	ocrresult,prob = clstm_ocr(img)
	print ocrresult, prob
