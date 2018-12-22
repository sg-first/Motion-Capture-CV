import numpy as np

# 参数为：待搜索模型、搜索范围、训练数据、验证数据（x），验证损失阈值（大于阈值不进行predict验证）、（训练和验证的）batch大小
def init_search(model,Range,TrainData,x_test,tho=99999,batch_size=10):
    for seed in range(Range[0],Range[1]):
        np.random.seed(seed)
        model.fit(TrainData[0], TrainData[1], epochs=1000, batch_size=batch_size)
        evalloss=model.evaluate(x, y, batch_size=batch_size)
        print(evalloss)
        if(evalloss>tho):
            continue
        else: # tho需要合理设置，然后到这基本就是行了
            y_test=model.predict(x_test, batch_size=len(x_test))
            for a in y_test:
                if a!=0:
                    print(y_test)
                    print(seed)
                    return seed
