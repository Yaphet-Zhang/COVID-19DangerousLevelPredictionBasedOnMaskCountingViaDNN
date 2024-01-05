from keras.models import load_model
import cv2
import numpy as np
import time

# load AdaBoost model 
face_detector=cv2.CascadeClassifier('./utils/haarcascade_frontalface_default.xml')
# load CNN model 
model = load_model(r'./weight/cnn_050.hdf5')
##### parameters #####
# 0=no mask=red, 1=mask=green  
labels_dict={0:'NO MASK',1:'MASK OK'}
color_dict={0:(0,0,255),1:(0,255,0)}
rectangle_thickness=2 # will be filled when -1
##### parameters #####


cap=cv2.VideoCapture(0)
while(True):
    start=time.time()
    ret,img=cap.read()
    #img=cv2.imread(r'./3.jpg') # read 1 image    

    # RGB image --> grayscale (speed up face detection)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # detect faces
    faces=face_detector.detectMultiScale(gray,1.3,5)  

    # return the face position information (left top x, left top y, w, h)
    for (x,y,w,h) in faces:
        # cut the face rectangle (left top y:right bottom y, left top x:right bottom x)   
        face_img=gray[y:y+h,x:x+w]
        # resize to 100×100    
        resized=cv2.resize(face_img,(100,100))
        # normalization: [0,255] --> [0,1]         
        normalized=resized/255.0
        # reshape --> [1,100,100,1] for CNN input
        reshaped=np.reshape(normalized,(1,100,100,1))
        # prediction value by CNN (e.g.[0.852,0.136])
        result=model.predict(reshaped)
        # prediction label (e.g.[0.852,0.136] --> 0)
        label=np.argmax(result,axis=1)[0]
        # draw face rectangle ((left top x,left top y),(right bottom x,right bottom y))
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],rectangle_thickness)
        # draw mask or no mask rectangle
        cv2.rectangle(img,(x,y-30),(x+w,y),color_dict[label],-1)
        # write mask or no mask and confidence score　
        cv2.putText(img,labels_dict[label]+':'+str('%.2f'%np.max(result)),(x, y-10),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)
       
    # show frame    
    cv2.imshow('Demo',img)
    key=cv2.waitKey(1)
    if key==27:
        break    
    
    end=time.time()
    print(end-start)
cv2.destroyAllWindows()
cap.release()

'''
# show 1 image 
cv2.imwrite('./111.jpg', img)
cv2.namedWindow('Demo',0)
cv2.resizeWindow('Demo',640,480)
cv2.imshow('Demo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''
