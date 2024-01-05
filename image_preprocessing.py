import cv2
import os
import numpy as np
from keras.utils import np_utils

data_path='dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels)) 

print('-----------')
print('categories,label:',label_dict)

##### parameters #####
img_size=100 # resize to 100*100 
##### parameters #####
data=[] # save images
target=[] # save labels

# count mask & no mask dataset number
count_no_mask=0
count_mask=0
c=0
for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)
    c+=1
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            # RGB image --> grayscale
            gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)           
            # grayscale --> 100x100
            resized=cv2.resize(gray,(img_size,img_size))
            # append image --> data list
            data.append(resized) 
            # append label --> target list
            target.append(label_dict[category])
            
            # count mask & no mask dataset number
            if c==1:
                count_no_mask+=1
            else:
                count_mask+=1
        
        
        except Exception as e:
            #if any exception rasied, the exception will be printed here. And pass to the next image
            print('Exception:',e)

print('-----------')            
print('len(data):',len(data))
print('data[0].shape:',data[0].shape)
print('len(target):',len(target))
print('target[0]:',target[0])
print('-----------')
print('len(no mask)',count_no_mask)
print('len(mask)',count_mask)

# normalization: [0,255] --> [0,1] 
data=np.array(data)/255.0
# reshape --> [1,100,100,1] for CNN input
data=np.reshape(data,(data.shape[0],img_size,img_size,1))
target=np.array(target)
# one-hot: 0 --> [1,0], 1 --> [0,1]
new_target=np_utils.to_categorical(target)

# write ndarray to .npy file
np.save('./utils/data',data)
np.save('./utils/target',new_target)
