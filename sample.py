import cv2
import numpy
 
img1 = cv2.imread("images/bloodcell.webp")
img2 = cv2.imread("images/bloodcell.webp")
 
img = numpy.concatenate((img1,img2),axis=1)
 
cv2.imshow("Concatenated Images",img)
 
cv2.waitKey(0)
 
cv2.destroyAllWindows()
