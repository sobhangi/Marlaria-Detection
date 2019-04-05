import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
image = cv.imread('sample.jpg')

#Gray scaling of images
gray_image=cv.cvtColor(image,cv.COLOR_BGR2GRAY)

plt.subplot(121),plt.imshow(image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(gray_image,cmap = 'gray')
plt.title('gray scale image'), plt.xticks([]), plt.yticks([])
plt.show()

#histogram equalization
equ = cv.equalizeHist(gray_image) 
res = np.hstack((gray_image,equ))
cv.imwrite('res.png',res)
plt.subplot(121),plt.imshow(gray_image,cmap = 'gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(equ,cmap = 'gray')
plt.title('contrast image'), plt.xticks([]), plt.yticks([])
plt.show()

image.shape

from scipy import ndimage
from sklearn import cluster
x, y, z = image.shape
image_2d = image.reshape(x*y, z)
image_2d.shape
kmeans_cluster = cluster.KMeans(n_clusters=7)
kmeans_cluster.fit(image_2d)
cluster_centers = kmeans_cluster.cluster_centers_
cluster_labels = kmeans_cluster.labels_

plt.figure(figsize = (15,8))
plt.imshow(cluster_centers[cluster_labels].reshape(x, y, z))