# Cognit

Cognit is a neural networking API and Framework built in python
cognit can be used with this command:

> [!IMPORTANT]
> Cognit has not been published to pip or any other packager, so its best
> if you use git to clone the repo and use it in your current directory


```
import cognit as cn

mnist = cn.deepflow.dataset.mnist
(x_train,y_train) = mnist.load()

model = cn.deepflow.sequential([
    
    cn.deepflow.layers.flatten(input_shape=(28,28)),
    cn.deepflow.layers.dense(12,activation="relu"),
    cn.deepflow.layers.dropout(rate=0.2,input_shape=(2,1)),
    cn.deepflow.layers.dense(10,1,activation="softmax")
])

model.train_data(optimiser="adam",X=x_train,y=y_train,layers_=model,loss_calc="sce",epochs=100)
```
 
