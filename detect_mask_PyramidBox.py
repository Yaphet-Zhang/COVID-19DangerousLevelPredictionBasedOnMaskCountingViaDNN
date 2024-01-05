from keras.models import load_model
import cv2
import numpy as np
import paddlehub as hub
import time

# load PyramidBox face detection model
model_face = hub.Module(name="pyramidbox_lite_mobile_mask")

# load CNN mask classification model
model_mask = load_model(r'./weight/cnn_050.hdf5')

##### parameters #####
# 0=no mask=red, 1=mask=green  
labels_dict = {0: 'NO MASK', 1: 'MASK OK'}
color_dict = {0: (0, 0, 255), 1: (0, 255, 0)}
rectangle_thickness = 7  # will be filled when -1
##### parameters #####


cap = cv2.VideoCapture(0)
while(True):
    start=time.time()
    ret,img=cap.read() # read live camera

    # img = cv2.imread(r'./sannomiya2.jpg') # read 1 image

    # return the face rectangle
    results = model_face.face_detection(data={"data": [img]})
    results = results[0]

    # RGB --> grayscale (for CNN preprocessing)
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # count the number of person
    sum_person_num=len(results['data'])
    mask_person_num=0
    no_mask_person_num=0

    for result in results['data']:
        confidence=result['confidence']
        left=int(result['left'])
        top=int(result['top'])
        right=int(result['right'])
        bottom=int(result['bottom'])

        # cut the face rectangle (top:bottom, left:right)
        face_img=gray[top:bottom,left:right]
        # resize to 100×100
        resized=cv2.resize(face_img,(100,100))
        # normalization: [0,255] --> [0,1]
        normalized=resized/255.0
        # reshape --> [1,100,100,1] for CNN input
        reshaped=np.reshape(normalized,(1,100,100,1))
        # prediction value by CNN (e.g.[0.852,0.136])
        result=model_mask.predict(reshaped)
        # prediction label (e.g.[0.852,0.136] --> 0)
        label=np.argmax(result,axis=1)[0]

        # count the number of mask person or no mask person
        if label==1:
            mask_person_num+=1
        else:
            no_mask_person_num+=1

        # draw face rectangle (left,top),(right,bottom)
        cv2.rectangle(img,(left,top),(right,bottom),color_dict[label],rectangle_thickness)
        # draw mask or no mask rectangle
        #cv2.rectangle(img,(left,top-30),(right,bottom),color_dict[label],-1)
        # write mask or no mask and confidence score
        cv2.putText(img,labels_dict[label]+':'+str('%.2f'%np.max(result*100))+'%',(left,top-10),cv2.FONT_HERSHEY_TRIPLEX,0.8,(255,255,255),1)

    # write the number of mask person or no mask person
    cv2.putText(img,'TOTAL:'+str(sum_person_num),(170,330),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(255,255,255),2)
    cv2.putText(img,'MASK:'+str(mask_person_num),(170,350),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,255,0),2)
    cv2.putText(img,'NO MASK:'+str(no_mask_person_num),(170,370),cv2.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),2)

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
cv2.imwrite('./sannomiya2_result.jpg', img)
cv2.namedWindow('Demo',0)
cv2.resizeWindow('Demo',640,480)
cv2.imshow('Demo',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''