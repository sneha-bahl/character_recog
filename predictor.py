# Import the modules
import cv2
from sklearn.externals import joblib
import numpy as np
import sys, getopt
from keras.models import load_model
from matplotlib import pyplot as plt
from PIL import Image
from resizeimage import resizeimage

def predict(input_image_name):
    fd_img = open("static/" + input_image_name, 'rb')            #pic path
    img = Image.open(fd_img)
    img = resizeimage.resize_thumbnail(img, [400, 500])
    resized_image = "static/resized_" + input_image_name
    img.save(resized_image, img.format)  #pic path
    fd_img.close()

    # Load the Keras CNN trained model
    model = load_model('model.h5')

    ################# NEW Algorithm with Adaptive Threshold #########################################
    # Read image in grayscale mode
    im = cv2.imread(resized_image)
    img = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("now", img)

    # Median Blur and Gaussian Blur to remove Noise
    img = cv2.medianBlur(img,5)
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive Threshold for handling lightning
    im_th = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,5)
    #cv2.imshow("Threshold Image",im_th)
    kernel = np.ones((1,1),np.uint8)
    im_th = cv2.dilate(im_th,kernel,iterations = 4)
    ##################################################################################################

    # Find contours in the image
    im2,ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Get rectangles contains each contour
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    i=0
    # For each rectangular region, predict using cnn model
    for rect in rects:
        # Draw the rectangles
        print("Rect",rect[0], rect[1], rect[2], rect[3])

        cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 3)
        # Make the rectangular region around the digit
        leng = int(rect[3] * 1.6)
        pt1 = int(rect[1] + rect[3] // 2 - leng // 2)
        pt2 = int(rect[0] + rect[2] // 2 - leng // 2)
        print("Point",pt1, pt2)
        #print(i,leng,pt1,pt2)
        i=i+1
        #roi = im_th[pt1:pt1+leng, pt2:pt2+leng]
        roi = im_th[rect[1]:rect[1] + rect[3], rect[0]:rect[0] + rect[2]]
        print(roi.shape)
        cv2.imwrite("static/strip" + str(i) + ".jpg", roi)

        if (roi.shape[0] < 1  or roi.shape[1] < 1):
            continue
        # Resize the image
        roi = cv2.resize(roi, (50, 50),interpolation=cv2.INTER_AREA)
        # Input for CNN Model
        roi = roi[np.newaxis,:,:,np.newaxis]

        # Input for Feed Forward Model
        # roi = roi.flatten()
        # roi = roi[np.newaxis]
        nbr = model.predict_classes(roi,verbose=0)
        nb = int(nbr[0])
        #print nb
        if nb<10:
            c=nb
        elif nb<36:
            c=chr(nb+55)
        else:
            c=chr(nb+61)
        cv2.putText(im, str(c), (int(rect[0]), int(rect[1])),cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255), 3)
    result_img_name = "result_" + input_image_name
    cv2.imwrite("static/" + result_img_name, im)
    return result_img_name
