import os
import shutil
fname='test.txt'

with open(fname) as f:
	list_files=[]
	for line in f:
	    content =str(line)
	    index=content.find(':')
	    content=content[:index]
	    list_files.append(content)
for file_name in list_files:
	base=os.path.basename(file_name)
	dirname=os.path.dirname(file_name)
	index=base.find('.')
	base=base[:index]
	base1=base+".bin.png"
	base1=os.path.join(dirname,base1)
	base2=base+".gt.txt"
	base2=os.path.join(dirname,base2)
	shutil.copy2(base1, 'test')
	shutil.copy2(base2, 'test')
print list_files