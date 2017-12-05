import requests

#http://docs.python-requests.org/en/latest/user/quickstart/#post-a-multipart-encoded-file

url = "http://localhost:5000/ocr/"
fin = open('tmp.png', 'rb')
files = {'file': fin}
try:
  r = requests.post(url, files=files)
  print r.text
finally:
	fin.close()