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

a=keras.layers.Dense(2, input_shape=(9,),activation='sigmoid')
b=keras.layers.Dense(2, input_shape=(9,),activation='sigmoid')
c=keras.layers.Dense(2, input_shape=(9,),activation='sigmoid')
d=keras.layers.Dense(2, input_shape=(9,),activation='sigmoid')

model2=a(model1)
model3=b(model1)
model4=c(model1)
model5=d(model1)

e=keras.models.Model(inputs=input,outputs=[model2,model3,model4,model5])
optimizer = keras.optimizers.SGD(lr=0.1, momentum=0)


e.compile(optimizer = optimizer,loss='binary_crossentropy')
y=[data.y_armL,data.y_armR,data.y_legL,data.y_legR]
grid.grid.init_search(e,[400,2790],[data.x,y],[data.x,y],data.x,0.1*1.6,-1,True)
print(e.evaluate(data.x,y, batch_size=2))
