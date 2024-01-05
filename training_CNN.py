import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.layers import Conv2D,MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.callbacks import ModelCheckpoint

#　load the ndarray (data & target)
data=np.load('./utils/data.npy') # [N images,100 width,100 height,1)
target=np.load('./utils/target.npy') # (N iamges,2)

# MODEL
model=Sequential()
# Conv1 + ReLU
model.add(Conv2D(64,(3,3),dilation_rate=(2,2),input_shape=data.shape[1:],kernel_initializer='he_normal'))
model.add(Activation('relu'))
# Conv2 + ReLU + Max Pooling
model.add(Conv2D(128,(3,3),dilation_rate=(2,2),kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# Conv3 + ReLU
model.add(Conv2D(256,(3,3),dilation_rate=(2,2),kernel_initializer='he_normal'))
model.add(Activation('relu'))
# Conv4 + ReLU + Max Pooling
model.add(Conv2D(256,(3,3),dilation_rate=(2,2),kernel_initializer='he_normal'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
# flatten C×W×H to 1-D
model.add(Flatten())
# Dropout 
model.add(Dropout(0.5))
# linear1 + ReLU
model.add(Dense(50,activation='relu',kernel_initializer='he_normal'))
# linear2 + Softmax
model.add(Dense(2,activation='softmax',kernel_initializer='he_normal'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

# split dataset
##### parameters #####
test_size=0.1
##### parameters #####
train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=test_size)

##### parameters #####
batch_size=32
epochs=50
validation_split=0.2
model_weight_path='./weight/cnn_{epoch:03d}.hdf5'
##### parameters #####

# checkpoint & history during training
checkpoint = ModelCheckpoint(model_weight_path,monitor='val_loss',verbose=0,save_best_only=False,mode='auto')
history=model.fit(train_data,train_target,batch_size=batch_size,epochs=epochs,callbacks=[checkpoint],validation_split=validation_split)

# visualize loss
plt.figure(1)
plt.plot(history.history['loss'],'r',label='training loss')
plt.plot(history.history['val_loss'],'b',label='validation loss')
plt.xlabel('epochs')
plt.ylabel('loss')
plt.legend()
plt.savefig('./weight/cnn_loss'+str(epochs)+'.png')
plt.show()

# visualize accuracy
plt.figure(2) 
plt.plot(history.history['accuracy'],'r',label='training accuracy')
plt.plot(history.history['val_accuracy'],'b',label='validation accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.legend()
plt.savefig('./weight/cnn_acc'+str(epochs)+'.png')
plt.show()

# evaluate the model by test set 
evaluation=model.evaluate(test_data,test_target)
print('-----------------')
print('loss:',evaluation[0],'accuracy:',evaluation[1])