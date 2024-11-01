from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QLabel, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap
from PIL import Image
import pandas as pd
import numpy as np 
import sys
import cv2
from widget import Ui_Dialog
import math
#detect the photo has been uploaded or not
photo1_upload = False
photo2_upload = False
#import photo
def import_photo():
    global photo1_upload, photo2_upload
    #this is for loading the image for usage
    file_name, _ = QFileDialog.getOpenFileName(None, 'Open Image', '.')
    ui.image = cv2.imread(file_name)  # read the image
    if ui.image is None:#make sure the image is in right format
        print("Error: we can only upload *.png, *.jpg, *.bmp.")
    else:
        if(ui.Load_Image_1.clicked):
            photo1_upload = True
        if(ui.Load_image_2.clicked):
            photo2_upload = True
        print(f"image has been uploaded")

#Q1
def separate_and_print():
    if(photo1_upload):
        b, g, r = cv2.split(ui.image)
        zeros = np.zeros(b.shape, dtype=np.uint8)
        # break all the color into 3 channel
        b_image = cv2.merge([b, zeros, zeros]) 
        g_image = cv2.merge([zeros, g, zeros]) 
        r_image = cv2.merge([zeros, zeros, r])
        cv2.imshow('b_image',b_image)
        cv2.imshow('g_image',g_image)
        cv2.imshow('r_image',r_image)
    else:
        print("photo haven't been uploaded")

def Grayscale():
    if(photo1_upload):
        cv_gray = cv2.cvtColor(ui.image,cv2.COLOR_BGR2GRAY)
        cv2.imshow('original grayscale',cv_gray)
        b, g, r = cv2.split(ui.image)
        avg_gray = (b/3+g/3+r/3).astype(np.uint8)
        cv2.imshow('average grayscale',avg_gray)
    else:
        print("photo haven't been uploaded")

def ext():
    if(photo1_upload):
        hsv_image = cv2.cvtColor(ui.image, cv2.COLOR_BGR2HSV)
        #15~85 can't totally get rid of green and yellow
        #so i choose 10~113, it can almost include all the green and yellow
        mask = cv2.inRange(hsv_image,np.array([10,0,0]),np.array([113,255,255]))
        mask_inverse = cv2.bitwise_not(mask)
        extracted_image = cv2.bitwise_and(ui.image,hsv_image, mask=mask_inverse)
        cv2.imshow("yellow green mask",mask)
        cv2.imshow("extracted picture",extracted_image)
    else:
        print("photo haven't been uploaded")

#Q2
def G_blur_popwindow():
    if(photo1_upload):
        cv2.namedWindow('G Blur')
        cv2.createTrackbar('m size', 'G Blur', 0, 5, apply_gaussian_blur)
        #cv2.createTrackbar('trackbar name', 'window's name', min, max, fn)

        apply_gaussian_blur(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("photo haven't been uploaded")

def apply_gaussian_blur(x):
    m = cv2.getTrackbarPos('m size', 'G Blur')
    #cv2.setTrackbarPos('trackbar name', 'window's name')
    kernel = 2 * m + 1  #(2m+1, 2m+1)
    Sigma = ((kernel-1)/2 -1)*0.3+0.8
    blurred_image = cv2.GaussianBlur(ui.image, (kernel, kernel), Sigma,Sigma)
    cv2.imshow('G Blur', blurred_image)

def B_blur_popwindow():
    if(photo1_upload):
        cv2.namedWindow('B Blur')
        #cv2.createTrackbar('trackbar name', 'window's name', min, max, fn)
        cv2.createTrackbar('m size', 'B Blur', 0, 5, apply_bilateral_blur)

        apply_bilateral_blur(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("photo haven't been uploaded")

def apply_bilateral_blur(x):
    m = cv2.getTrackbarPos('m size', 'B Blur')
    #cv2.setTrackbarPos('trackbar name', 'window's name')
    kernel = (2 * m + 1)  #diameter
    
    blurred_image = cv2.bilateralFilter(ui.image, kernel,75,75)
    # according to the below website
    # https://www.geeksforgeeks.org/python-bilateral-filtering/
    cv2.imshow('B Blur', blurred_image)

def M_blur_popwindow():
    if(photo1_upload):
        cv2.namedWindow('M Blur')
        #cv2.createTrackbar('trackbar name', 'window's name', min, max, fn)
        cv2.createTrackbar('m size', 'M Blur', 0, 5, apply_median_blur)

        apply_median_blur(1)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("photo haven't been uploaded")

def apply_median_blur(x):
    m = cv2.getTrackbarPos('m size', 'M Blur')
    #cv2.setTrackbarPos('trackbar name', 'window's name')
    kernel = (2 * m + 1)  #diameter
    blurred_image = cv2.medianBlur(ui.image, kernel)
    cv2.imshow('M Blur', blurred_image)

#Q3
def sobel_x():
    if(photo1_upload):  
        gray = cv2.cvtColor(ui.image, cv2.COLOR_BGR2GRAY)
        Sigma = ((3-1)/2 -1)*0.3+0.8
        blur = cv2.GaussianBlur(gray, (3, 3), Sigma,Sigma)
        #create matrix that is full of zero
        x_photo=np.zeros_like(blur).astype(np.int32)
        #ready to calculate the pixal 
        width,height = x_photo.shape
        # Put in the array
        x_photo_filter=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        #run every pixal
        for w in range(width):
            for h in range(height):
                for i in range(3):
                    for j in range(3):
                        if(w+i>=0 and w+i-1<width and h+j-1>=0 and h+j-1<height):
                            #every pixal needs to be multiply by the matrix
                            x_photo[w][h] += blur[w+i-1][h+j-1] * x_photo_filter[i][j]
        
        # restrict the value in 0 ~ 255
        x_photo = np.where(x_photo < 0,x_photo*-1,x_photo)
        x_photo = np.where(x_photo>255,255,x_photo)
        cv2.imshow("Sobel_X photo",x_photo.astype(np.uint8))
        print("photo haven't been uploaded")

def sobel_y():#same as sobel_x
    if(photo1_upload):  
        gray = cv2.cvtColor(ui.image, cv2.COLOR_BGR2GRAY)
        Sigma = ((3-1)/2 -1)*0.3+0.8
        blur = cv2.GaussianBlur(gray, (3, 3), Sigma,Sigma)
        y_photo=np.zeros_like(blur).astype(np.int32)
        width,height=y_photo.shape
        y_photo_filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        for w in range(width):
            for h in range(height):
                for i in range(3):
                    for j in range(3):
                        if(w+i>=0 and w+i-1<width and h+j-1>=0 and h+j-1<height):
                            y_photo[w][h] += blur[w+i-1][h+j-1] * y_photo_filter[i][j]
        
        y_photo = np.where(y_photo < 0,y_photo*-1,y_photo)
        y_photo = np.where(y_photo>255,255,y_photo)
        cv2.imshow("Sobel_Y photo",y_photo.astype(np.uint8))
    else:
        print("photo haven't been uploaded")

def comb():
    if(photo1_upload): 
        gray = cv2.cvtColor(ui.image, cv2.COLOR_BGR2GRAY)
        Sigma = ((3-1)/2 -1)*0.3+0.8
        blur = cv2.GaussianBlur(gray, (3, 3), Sigma,Sigma)
        
        x_photo_filter=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        x_photo=np.zeros_like(blur).astype(np.int32)
        width,height = x_photo.shape
        for w in range(width):
            for h in range(height):
                for i in range(3):
                    for j in range(3):
                        if(w+i>=0 and w+i-1<width and h+j-1>=0 and h+j-1<height):
                            x_photo[w][h] += blur[w+i-1][h+j-1]*x_photo_filter[i][j]
        
        x_photo = np.where(x_photo < 0,x_photo*-1,x_photo)
        x_photo = np.where(x_photo>255,255,x_photo)

        y_photo_filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
        y_photo=np.zeros_like(blur).astype(np.int32)
        width,height=y_photo.shape
        for w in range(width):
            for h in range(height):
                for i in range(3):
                    for j in range(3):
                        if(w+i>=0 and w+i-1<width and h+j-1>=0 and h+j-1<height):
                            y_photo[w][h] += blur[w+i-1][h+j-1]*y_photo_filter[i][j]

        y_photo = np.where(y_photo < 0,y_photo*-1,y_photo)
        y_photo = np.where(y_photo>255,255,y_photo)

        #combine photo image by using formula
        combine  = np.zeros_like(y_photo).astype(np.int32)
        for w in range(width):
            for h in range(height):
                combine[w][h] = (x_photo[w][h]**2 + y_photo[w][h]**2)**0.5
        
        _,result = cv2.threshold(combine.astype(np.uint8), 128, 255, cv2.THRESH_BINARY)
        _,result2 = cv2.threshold(combine.astype(np.uint8), 28, 255, cv2.THRESH_BINARY)
        cv2.imshow("square photo", combine.astype(np.uint8))
        cv2.imshow("threshold=128",result.astype(np.uint8))
        cv2.imshow("threshold=28",result2.astype(np.uint8))
    else:
        print("photo haven't been uploaded")

def gradient_angle():
    if (photo1_upload):
        gray = cv2.cvtColor(ui.image, cv2.COLOR_BGR2GRAY)
        Sigma = ((3-1)/2 -1)*0.3+0.8
        blur = cv2.GaussianBlur(gray, (3, 3), Sigma,Sigma)
        
        x_photo_filter=np.array([[1,0,-1],[2,0,-2],[1,0,-1]])
        y_photo_filter=np.array([[1,2,1],[0,0,0],[-1,-2,-1]])

        x_photo=np.zeros_like(blur).astype(np.int32)
        y_photo=np.zeros_like(blur).astype(np.int32)
        
        width,height = x_photo.shape
        for w in range(width):
            for h in range(height):
                for i in range(3):
                    for j in range(3):
                        if(w+i>=0 and w+i-1<width and h+j-1>=0 and h+j-1<height):
                            x_photo[w][h] += blur[w+i-1][h+j-1]*x_photo_filter[i][j]
                            y_photo[w][h] += blur[w+i-1][h+j-1]*y_photo_filter[i][j]

        magnitude = np.zeros_like(y_photo).astype(np.int32)
        #create magnitude to calculate the combine pic
        for w in range(width):
            for h in range(height):
                magnitude[w][h] = (x_photo[w][h]**2 + y_photo[w][h]**2)**0.5

        #calculate angle
        angle = (np.arctan2(y_photo, x_photo) * 180 / math.pi) % 360

        #create mask 
        mask1 = cv2.inRange(angle, 170 , 190)  # 檢測170-190
        mask2 = cv2.inRange(angle, 260 , 280)  # 檢測260-280

        #do the mask
        masked_result1 = cv2.bitwise_and(magnitude, magnitude, mask=mask1)
        masked_result2 = cv2.bitwise_and(magnitude, magnitude, mask=mask2)
        masked_result1=masked_result1.astype(np.uint8)
        masked_result2=masked_result2.astype(np.uint8)       

        cv2.imshow("angle 1 (170~190)", masked_result1)  
        cv2.imshow("angle 2 (260~280)", masked_result2)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Photo hasn't been uploaded")

#Q4
def transform():
    if (photo1_upload):
        angle = float(ui.Rotation.text())  
        scale = float(ui.scaling.text())  
        tx = float(ui.Tx.text()) 
        ty = float(ui.Ty.text()) 

        angle_rad = math.radians(360-angle)

        rotation_matrix=np.array([[scale * math.cos(angle_rad),-math.sin(angle_rad), tx],
                         [math.sin(angle_rad), scale * math.cos(angle_rad), ty],[0,0,1]])
        affine_matrix = rotation_matrix[:2,:3]
        result = cv2.warpAffine(ui.image, affine_matrix, (1920, 1080))
        cv2.imshow("result",result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    ui.Load_Image_1.clicked.connect(import_photo)
    ui.Load_image_2.clicked.connect(import_photo)
    ui.color_separation.clicked.connect(separate_and_print)
    ui.color_transformation.clicked.connect(Grayscale)
    ui.color_extraction.clicked.connect(ext)
    ui.gaussian_filter.clicked.connect(G_blur_popwindow)
    ui.bilateral_filter.clicked.connect(B_blur_popwindow)
    ui.median_fliter.clicked.connect(M_blur_popwindow)
    ui.sobel_x.clicked.connect(sobel_x)
    ui.sobel_y.clicked.connect(sobel_y)
    ui.combination_and_threshold.clicked.connect(comb)
    ui.gradiant_angle.clicked.connect(gradient_angle)
    ui.pushButton.clicked.connect(transform)
    Dialog.show()
    sys.exit(app.exec_())