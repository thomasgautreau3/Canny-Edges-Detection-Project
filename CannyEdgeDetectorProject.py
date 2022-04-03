#code from https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from scipy import ndimage
import math
import time
import os

startTime = time.time()
directory = "C:/Users/thoma/cs2704/BIPED/BIPED/edges/imgs/train/rgbr/real"
iterations = 0
for image in os.listdir(directory):

    print(iterations)
    iterations+=1
    #---------------------------------Greyscale conversion----------------------------------------
    #getting image
    # filename = os.fsdecode(image)
    # img = np.array(Image.open(image)).astype(np.uint8)
    filename = directory + '/' + image
    img = np.array(Image.open(filename)).astype(np.uint8)

    # plt.imshow(img)
    # plt.show()

    #grayscale
    R, G, B = img[:,:,0], img[:,:,1], img[:,:,2] #for each pixel
    imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B
    # plt.imshow(imgGray, cmap='gray')
    # plt.show()

    #----------------------------------------------------------------------------------------------

    #-------------------------------Noise Reduction/Gaussian blur----------------------------------
    # If the size of the kernel is increased, the blur will increase.
    # if the sigma is increased, the blur will also increase
    def gaussian_kernel(size, sigma=1.4):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1 / (2.0 * np.pi * sigma**2)
        g =  np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
        return g

    kernel = gaussian_kernel(5)
    blurImg = ndimage.convolve(imgGray, kernel)
    # plt.imshow(blurImg, cmap='gray')
    # plt.show()

    #----------------------------------------------------------------------------------------------

    #------------------------------------Gradient/Sobel calculation---------------------------------

    #gradient X and Y
    gx = np.array([[-1,0,1], [-2, 0, 2], [-1, 0, 1]], np.float32) #apparently it matters which index the -1/-2's are for this to work properly
    gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

    imgX = ndimage.convolve(blurImg, gx)
    imgY = ndimage.convolve(blurImg, gy)

    gradientMagXY = np.hypot(imgX, imgY) #getting the hypotenuse NEED TO CHANGE THIS...
    orientationImg = np.arctan2(imgY, imgX)

    sobelImg = gradientMagXY/gradientMagXY.max() * 255
    #G/G.max() * 255 # why do we want to divide each pixel in G by it's max pixel value?
    #maybe check what G.max() would give for more insight/

    #-----------------------------------------------------------------------------------------------

    #------------------------------------Non-Maximum suppression-------------------------------------

    #create a matrix of same size as gradient matrix and init to all 0's
    rows, cols = sobelImg.shape
    nonMaxSupr = np.zeros((rows, cols), dtype=np.int32)

    #convert to degrees
    edgeAngle = orientationImg * 180.0/np.pi #(or could do 180.0/np.pi)

    #This will give us only positive angles because its the same thing as neg since we will check
    #both sides of the cell
    edgeAngle[edgeAngle < 0] += 180 

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            try:
            
                pixel1 = 255 #The pixel that will be the top, top right, or right
                pixel2 = 255 #The pixel that will be the bottom, bottom left, or left

                if ((edgeAngle[i, j] >= 0) and (edgeAngle[i,j] < 22.5)) or ((edgeAngle[i,j] <= 180) and (edgeAngle[i,j] > 157.5)):
                    pixel1 = sobelImg[i, j+1] 
                    pixel2 = sobelImg[i, j-1]

                elif ((edgeAngle[i,j] >= 22.5) and (edgeAngle[i,j] < 67.5)):
                    pixel1 = sobelImg[i-1, j+1] 
                    pixel2 = sobelImg[i+1, j-1]

                elif ((edgeAngle[i,j] >= 67.5) and (edgeAngle[i,j] < 112.5)):
                    pixel1 = sobelImg[i-1, j] 
                    pixel2 = sobelImg[i+1, j]

                elif ((edgeAngle[i,j] >= 112.5) and (edgeAngle[i,j] < 157.5)):
                    pixel1 = sobelImg[i-1, j-1] 
                    pixel2 = sobelImg[i+1, j+1]

                if (pixel1 <= sobelImg[i, j]) and (pixel2 <= sobelImg[i, j]):
                    nonMaxSupr[i, j] = sobelImg[i, j] #or can try 255 to see what it yields (does something cool :)
                else:
                    nonMaxSupr[i, j] = 0
                    
            except IndexError as e:
                pass

    #---------------------------------------------------------------------------------------------------------------

    #------------------------------------------Thresholding-------------------------------------------------------

    highThresholdRatio = 0.09
    lowThresholdRatio = 0.05

    highThreshold = img.max() * highThresholdRatio
    lowThreshold = highThreshold * lowThresholdRatio

    row, col = nonMaxSupr.shape
    result = np.zeros((row, col), dtype=np.int32)

    strongEdge_i, strongEdge_j = np.where(nonMaxSupr >= highThreshold)
    zero_i, zero_j = np.where(nonMaxSupr < lowThreshold)

    weakEdge_i, weakEdge_j = np.where((nonMaxSupr >= lowThreshold) & (nonMaxSupr < highThreshold))

    result[strongEdge_i, strongEdge_j] = np.int32(255)
    result[weakEdge_i, weakEdge_j] = np.int32(100)

    #---------------------------------------------------------------------------------------------------------------

    #---------------------------------------------Hysteresis Edge Tracking------------------------------------------

    #Turning a weak egde into a strong one if they have at least one neighbouring edge that is strong
    strong = np.int32(255)
    weak = np.int32(100)

    for i in range(1,rows-1):
        for j in range(1,cols-1):
            if result[i,j] == weak:
            
                if ((result[i, j+1] == strong) or (result[i-1, j+1] == strong) or (result[i-1, j+1] == strong)
                    or (result[i-1, j] == strong) or (result[i+1, j] == strong) or (result[i, j-1] == strong)
                    or (result[i+1, j-1] == strong) or (result[i-1, j-1] == strong)):
                        result[i, j] = strong
                else: 
                    result[i,j] = np.int32(0)

    if iterations == 10:
        break
    #-----------------------------------------------------------------------------------------------------------------


print(time.time()-startTime)

