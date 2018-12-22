import numpy as np

# 参数为：待搜索模型、搜索范围、训练数据、验证数据、预测数据（x），验证损失阈值（大于阈值不进行predict验证）、（训练和验证的）batch大小
def init_search(model,Range,TrainData,testData,x_pre,tho=99999,batch_size=-1):
    if batch_size==-1:
        batch_size=len(TrainData[0])
        
    for seed in range(Range[0],Range[1]):
        np.random.seed(seed)
        model.fit(TrainData[0], TrainData[1], epochs=1000, batch_size=batch_size)
        evalloss=model.evaluate(testData[0], testData[1], batch_size=batch_size)
        print(evalloss)
        if(np.mean(evalloss)>tho):
            continue
        else: # tho需要合理设置，然后到这基本就是行了
            y_test=model.predict(x_pre, batch_size=len(x_pre))
            print(y_test)
            print(seed)
            return seed
