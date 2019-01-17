import keras
import data
import numpy as np
import grid.grid
from keras import backend as K


input=keras.layers.Input((4,4))

model=keras.Sequential()
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(12, input_shape=(16,),activation='relu')) # 添加全连接层
model.add(keras.layers.Dense(10, input_shape=(12,),activation='relu'))
model.add(keras.layers.Dense(9, input_shape=(10,),activation='relu'))

model1=model(input)

out_arm_L=keras.layers.Dense(2, input_shape=(9,), activation='softmax')(model1)
out_arm_R=keras.layers.Dense(2, input_shape=(9,), activation='softmax')(model1)
out_leg_L=keras.layers.Dense(2, input_shape=(9,), activation='softmax')(model1)
out_leg_R=keras.layers.Dense(2, input_shape=(9,), activation='softmax')(model1)

def binary_crossentropy(args):
    y_true, y_pred=args
    out= K.mean(K.binary_crossentropy(y_pred, y_true), axis=-1)
    return  out

input1=keras.layers.Input(shape=(2,))
input2=keras.layers.Input(shape=(2,))
input3=keras.layers.Input(shape=(2,))
input4=keras.layers.Input(shape=(2,))
#计算损失函数层
loss_out1 = keras.layers.Lambda(binary_crossentropy, output_shape=(2,))([input1, out_arm_L]) # output_shape是函数返回值的shape，第一个维度是占位符
loss_out2 = keras.layers.Lambda(binary_crossentropy, output_shape=(2,))([input2, out_arm_R]) #
loss_out3 = keras.layers.Lambda(binary_crossentropy, output_shape=(2,))([input3, out_leg_L])
loss_out4 = keras.layers.Lambda(binary_crossentropy, output_shape=(2,))([input4, out_leg_R])

def loss(y_true,y_pred):
    return y_pred

y=[data.y_armL,data.y_armR,data.y_legL,data.y_legR]

optimizer = keras.optimizers.SGD(lr=0.1, momentum=0)

f=keras.models.Model(inputs=input,outputs=[out_arm_L,out_arm_R,out_leg_L,out_leg_R])
f.compile(optimizer=optimizer,loss=loss)

e=keras.models.Model(inputs=[input,input1,input2,input3,input4],outputs=[loss_out1,loss_out2,loss_out3,loss_out4])
e.compile(optimizer=optimizer,loss=loss)
grid.grid.init_search(e,[400,2790],[[data.x]+y,y],[[data.x]+y,y],[data.x]+y,tho=0.05,batch_size=-1,isMulOut=True)
#model.compile(optimizer='rmsprop',loss='hinge')
print(f.evaluate(data.x,y, batch_size=2))
x_test=data.x
print(f.predict(x_test, batch_size=7))