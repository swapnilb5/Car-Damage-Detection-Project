import numpy as np
import cv2
import os 

dir_path = os.getcwd()
print(dir_path)
width = 640
height = 640
dim = (width, height)

def main():
    for filename in os.listdir(dir_path):
        # If image are not .jpg image , chn]ange the line below to match the image type.
        if filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".JPG"):
            image = cv2.imread(filename)
            print(filename)
            resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
            cv2.imwrite(filename, resized)
            
    
    
    main()