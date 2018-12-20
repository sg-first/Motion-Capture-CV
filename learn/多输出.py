import keras
import data
import numpy as np

from keras import backend as K
model=keras.Sequential()
input=keras.layers.Input((4,4))
#model.add(keras.engine.input_layer.Input(shape=(4,4)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(12, input_shape=(16,),activation='relu')) # 添加全连接层
model.add(keras.layers.Dense(10, input_shape=(12,),activation='relu'))
model.add(keras.layers.Dense(9, input_shape=(10,),activation='softmax'))
a=keras.layers.Dense(8, input_shape=(9,),activation='softmax')
b=keras.layers.Dense(8, input_shape=(9,),activation='softmax')
c=keras.layers.Dense(8, input_shape=(9,),activation='softmax')
d=keras.layers.Dense(8, input_shape=(9,),activation='softmax')
model1=model(input)
model2=a(model1)
model3=b(model1)
model4=c(model1)
model5=d(model1)
e=keras.models.Model(inputs=model1,outputs=[model2,model3,model4,model5])
e.compile(optimizer='rmsprop',loss='hinge')
e.fit(data.x, [data.y_armL,data.y_armR,data.y_legL,data.y_legR],epochs=1000, batch_size=2)
print(e.evaluate(data.x,[data.y_armL,data.y_armR,data.y_legL,data.y_legR], batch_size=2))
print(e.predict(data.x, batch_size=2))