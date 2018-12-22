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
model.add(keras.layers.Dense(9, input_shape=(10,),activation='softmax'))

model1=model(input)

a=keras.layers.Dense(2, input_shape=(9,),activation='softmax')
b=keras.layers.Dense(2, input_shape=(9,),activation='softmax')
c=keras.layers.Dense(2, input_shape=(9,),activation='softmax')
d=keras.layers.Dense(2, input_shape=(9,),activation='softmax')

model2=a(model1)
model3=b(model1)
model4=c(model1)
model5=d(model1)

e=keras.models.Model(inputs=input,outputs=[model2,model3,model4,model5])
e.compile(optimizer='rmsprop',loss='hinge')
grid.grid.init_search(e,[0,1280],[data.x,[data.y_armL,data.y_armR,data.y_legL,data.y_legR]],[data.x,[data.y_armL,data.y_armR,data.y_legL,data.y_legR]],data.x,0.5,-1,True)
print(e.evaluate(data.x,[data.y_armL,data.y_armR,data.y_legL,data.y_legR], batch_size=2))
