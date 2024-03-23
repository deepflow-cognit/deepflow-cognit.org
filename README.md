# Cognit

Cognit is a neural networking API and Framework built in python
cognit can be used with this command:

```
import cognit as cn

mnist = cn.deepflow.dataset.mnist()
(trainX,trainY) = mnist.load()

cn.deepflow.sequential([
    cn.deepflow.layers.flatten(input_shape=(28,28)),
    cn.deepflow.layers.dense(12,1,activation="relu"),
    cn.deepflow.layers.dropout(0.2),
    cn.deepflow.layers.dense(10,1,activation="softmax")
])

cn.deepflow.train_data(optimiser="adam",X=trainX,y=trainY,loss_calc="sce",epochs=100)
```
