# Cognit

Cognit is a neural networking API and Framework built in python
cognit can be used with this command:

```
import cognit as cn
import numpy as np

dataset = cn.deepflow.dataset.mnist
(X_train,y_train), (X_train,y_train) = cn.deepflow.dataset.load()

model = cn.deepflow.sequential([
    cn.deepflow.layers.flatten(X=12),
    cn.deepflow.layers.dense(128,activation="relu"),
    cn.deepflow.layers.dense(10,activation="softmax")
])

cn.deepflow.train_data(X=X_train,y=y_train,optimiser="adam",epochs=100)
```
