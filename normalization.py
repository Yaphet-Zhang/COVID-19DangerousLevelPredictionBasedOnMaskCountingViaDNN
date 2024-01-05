# -*- coding: utf-8 -*-
"""
Created on Sun Jan 17 22:08:54 2021

@author: zhangbowen
"""

import cv2

img=cv2.imread('./fanbingbing.jpg')
normalized_img=img/180.0
cv2.imwrite('fanbingbing_img.jpg', img)
cv2.imwrite('fanbingbing_normalized.jpg', normalized_img)
cv2.imshow('img',img)
cv2.imshow('normalized',normalized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()