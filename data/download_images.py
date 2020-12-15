import urllib.request
import os
import argparse
import json

parser = argparse.ArgumentParser()

parser.add_argument('--input_json', required=True, help='enter json input')
parser.add_argument('--images_root', required=True, help='enter root directory')

args = parser.parse_args()
params = vars(args)
input_json = params['input_json']
images_root = params['images_root']

with open(input_json) as data_file:
	data = json.load(data_file)

image_urls = data['unique_img_train']
image_urls += data['unique_img_test']

# print(len(image_urls))

rejected = []
for i in image_urls:
	img = 'https://' + i
	img = img.replace('-','/')

	try:
		urllib.request.urlretrieve(img, images_root+i)
	except:
		try:
			img = img.replace('.jpg','.png')
			urllib.request.urlretrieve(img, images_root+i)
		except:
			rejected.append(i)
			continue

print("not downloaded : ")
print(rejected)
