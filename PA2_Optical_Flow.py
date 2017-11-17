# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 08:19:16 2017

@author: 0
"""
from scipy.misc import imresize
from scipy.signal import convolve,convolve2d
import scipy
from PIL import Image
import cv2
import numpy as np
img = cv2.imread("C://Users/0/Downloads/basketball1.png",0)
img2 = cv2.imread("C://Users/0/Downloads/basketball2.png",0)
#cv2.imshow('img',img)
#cv2.imshow('img2',img2)
k=(3,3)
print img
img = cv2.GaussianBlur(img, k, 1.5)
img2 = cv2.GaussianBlur(img2, k, 1.5)

cv2.imshow('img3',img)
#cv2.waitKey(10000)
cv2.destroyAllWindows()
imga=np.matrix(img)
imga2=np.matrix(img2)
#print imga
#img=Image.fromarray(imga)
#img.show()
height,width = imga.shape
#for x in range img(x,0):
print imga.shape
print height ,width
#    print x
#for y in height:
#    for x in width:
#        print '0'
#for y in range(height):
print imga
#imga[0,1]=imga[0,1]+1
#print imga
def fx(y,x):
    fx=(int(imga[y,x+1])-int(imga[y,x]))/1
    
    return fx
def fy(y,x):
    fy=(int(imga[y+1,x])-int(imga[y,x]))/1
    
    return fy
print fx(1,0),fy(0,4)
imga=imresize(imga,(240,320))
imga2=imresize(imga2,(240,320))
print imga,imga.shape,imga2,imga2.shape
u=np.zeros([240,320])
v=np.zeros([240,320])
w2=30
w=15
#for i in range(w2):
#    for y in range(w2):
#        
#    
#    print matrix
#matrix=np.zeros([w2,w2])
#
#for x in range(w,240-w):
#    
#    for y in range(w,320-w):
#        c=0
##        matrix[w,w]=x
#    print x,y
#print matrix
#def conv2(x, y, mode='same'):
#    return np.rot90(convolve2d(np.rot90(x, 2), np.rot90(y, 2), mode=mode), 2)
#print convolve2d(imga2,matrix,'valid')


'''
ft = scipy.signal.convolve2d(imga, 0.25 * np.ones((2,2))) + \
       scipy.signal.convolve2d(imga2, -0.25 * np.ones((2,2)))
       
#print ft
fx,fy=np.gradient(cv2.GaussianBlur(img, k, 1.5))
fx = fx[0:478, 0:638]  
fy = fy[0:478, 0:638]
ft = ft[0:478, 0:638]
#print fx,fy,ft
'''


'''
for i in range(w+1,480-w):
   for j in range(w+1,640-w):
      Ix = fx[i-w:i+w, j-w:j+w]
      Iy = fy[i-w:i+w, j-w:j+w]
      It = ft[i-w:i+w, j-w:j+w]
A = [Ix,Iy]
print fx,fy,ft
'''
#C=A.T*-It
#print C
#print curFx,curFy,curFt,U[0],U[1]
