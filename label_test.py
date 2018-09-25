import cv2
import pandas as pd
import pytesseract
import re

label=["GBB9032P","SLB9944P","SLP8772B","SKH5906G","SLZ300E","SLG9325A","SLU7747","SJY1928S","SLF4095B","SGK368Z","SJN7561J","SHA7822Z","SLG8804R","SMA2887K","SHC7520J","SLL8936S","SKC8287R","SKE5353S","SLM3669R","SJQ2807T","SLT3474Z","SLS7130U","SLN2539J","SKB9993J","SKQ9052X","SLQ6081U","SLJ8660Y","SLZ5663A","SLK7442L","SHB4493D","GBF5143M","SLM4222P","SLX8550H"]



for i in range(1,34):
	path='num2/num'+str(i)+'.jpg'
	
	print(path)
	img=cv2.imread(path)
	
	config=('-l eng --oem 1 --psm 3')
	
	text=pytesseract.image_to_string(img,config=config)
	
	text=re.sub('[^A-Za-z0-9]+', '', text)
	
	imagename=path[5:]
	raw_data={"Image_name":[imagename],"True_label":[label[i-1]],"image_label":[text]}
	
	
	column_name = ['Image_name', 'True_label','image_label']
	df = pd.DataFrame(raw_data, columns = column_name)
	df.to_csv('/home/katomaran/Desktop/Tesseract/label_tesseract.csv', mode='a', header=False)
	
	
