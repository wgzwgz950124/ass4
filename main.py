# -*- coding: utf-8 -*-
"""
Created on Fri Jan 10 21:32:53 2020

@author: gzw
"""

import kmeans as km
from sklearn.cluster import KMeans
from sklearn.externals import joblib


#if __name__ == '__main__':
#
#    image = cv2.imread('D://vip//ass4//camera.png',0)
#
#    k = 3
#    threshold = 1
#    labels = km.k_means(image, k, threshold)
#
#    plt.subplot(1, 2, 1)
#    plt.title("Soucre Image")
#    plt.imshow(image,cmap="gray")
#    plt.subplot(1, 2, 2)
#    plt.title("Segamenting Image with k-means\n" + "k=" + str(k) + "  threshold=" + str(threshold))
#    plt.imshow(labels)
#    plt.show()
#    print(np.unique(labels))
#    print(labels.shape)

import cv2
import matplotlib.pyplot as plt
import numpy as np


def seg_kmeans_gray():
    img = cv2.imread('D://vip//ass4//camera.png', cv2.IMREAD_GRAYSCALE)

    # 展平
    img_flat = img.reshape((img.shape[0] * img.shape[1], 1))
    img_flat = np.float32(img_flat)

    # 迭代参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TermCriteria_MAX_ITER, 100, 0.5)
    flags = cv2.KMEANS_RANDOM_CENTERS

    # 聚类
    compactness, labels, centers = cv2.kmeans(img_flat, 2, None, criteria, 100, flags)
    
    # 显示结果
    img_output = labels.reshape((img.shape[0], img.shape[1]))
    plt.subplot(121), plt.imshow(img, 'gray'), plt.title('input')
    plt.subplot(122), plt.imshow(img_output, 'gray'), plt.title('kmeans')
    plt.show()
    print(np.unique(labels))

if __name__ == '__main__':
    seg_kmeans_gray()