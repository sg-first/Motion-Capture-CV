import keras
import data
import numpy as np

from keras import backend as K
model=keras.Sequential()
#model.add(keras.engine.input_layer.Input(shape=(4,4)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(6, input_shape=(16,),activation='relu')) # 添加全连接层
model.add(keras.layers.Dense(4, input_shape=(6,),activation='relu'))
model.add(keras.layers.Dense(2, input_shape=(2,),activation='softmax'))
model.compile(optimizer='rmsprop',loss='hinge')
model.fit(data.x, data.y_armL,epochs=1000, batch_size=2)
print(model.evaluate(data.x,data.y_armL, batch_size=2))
print(model.predict(data.x, batch_size=2))
