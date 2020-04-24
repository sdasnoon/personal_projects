
#transforming image in Numpy  - This helps learn the basics of Image processing
# https://towardsdatascience.com/3-numpy-image-transformations-on-baby-yoda-c27c1409b411
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import os
path = os.path.dirname(os.path.abspath("__file__"))
img = plt.imread(path+'/resources/baby_yoda_1.jpeg')

#show my image
plt.imshow(img)
#create a copy of my image
copyImg = img.copy()

#get the average of each pixel
#we need axis=2 so that we take averages accross the 3rd axis (the color axis) rather than accross the rows or columns
average_pixel_values = np.mean(copyImg, axis=2, keepdims=True)

#create the grey image by "stacking" three copies of these grey values together
greyImage = np.concatenate([average_pixel_values]*3, axis=2)

#make sure to cast the final image back to integers
greyImage = greyImage.astype(int)

#show the final image
plt.imshow(greyImage)

# To reverse a list use : list_of_nums[::-1] - this is a shorthand for [0:len(list_of_nums),-1]

# Vertical Flip
#reverse all the rows in my image
vFlipImg = copyImg[::-1]

#show the image
plt.imshow(vFlipImg)

#Horizontal Flip
# [:,::-1]. Well, this is again just a shorthand for [0:numRows, 0:numCols:-1], which in words says “iterate overall my rows and iterate over all my columns in reverse order”

#reverse all the rows in my image
hFlipImg = copyImg[:,::-1]

#show the image
plt.imshow(hFlipImg)

#Image Blur Effect

#create a copy of my grey image and just get first layer (because all layers are same)
copyImg = greyImage.copy()[:,:,0]

#get number of rows, columns in our image
numRows,numCols = copyImg.shape

#define how big we want our blurring box to be, the bigger the blurrier
boxSize = 31

#get half the size of the box
halfBoxSize = int(boxSize/2)

#we can only blur pixels where the box can fit around them (i.e. the edges wont get blurred)
startRow = halfBoxSize
startCol = halfBoxSize

#loop over all the valid pixels in the image
for row in range(startRow, numRows-halfBoxSize):
    for col in range(startCol, numCols-halfBoxSize):

        #create the local box around a given pixel
        localPixels = greyImage[row-halfBoxSize:row+halfBoxSize+1, col-halfBoxSize:col+halfBoxSize+1][:,:,0]

        #take the mean of the pixels in the local box
        blurredValue = np.mean(localPixels)

        #set the new value at that pixel to be that blurred value
        copyImg[row,col] = blurredValue

#reshape the flat image to give it a third dimension of 1
copyImg = copyImg.reshape([numRows, numCols, 1])

#stack 3 copies of the transformed image together to create the blurred image
blurredImage = np.concatenate([copyImg]*3, axis=2)

plt.imshow(blurredImage)
