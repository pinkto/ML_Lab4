import numpy as np
import pandas as pd
from tensorflow.python.keras import layers
from tensorflow.python.keras import models


def calcNumpy(a, b, c):
    return np.logical_or(np.logical_and(a, np.logical_not(b)), np.logical_xor(c, b))


def calcElementWise(a, b, c):
    return np.array([((a[i] and not(b[i])) or (c[i] ^ b[i])) for i in range(0, len(a))])


a = [0, 0, 0, 0, 1, 1, 1, 1]
b = [0, 0, 1, 1, 0, 0, 1, 1]
c = [0, 1, 0, 1, 0, 1, 0, 1]


ansNumpy = calcNumpy(a, b, c)
ansElementWise = calcElementWise(a, b, c)

beforeLearning = pd.DataFrame(data={
    "a" : a,
    "b" : b,
    "c" : c,
    "numpy" : ansNumpy,
    "element-wise" : ansElementWise
})

beforeLearning.to_csv('beforeLearning.csv')

model = models.Sequential()
model.add(layers.Dense(16, activation='relu', input_shape=(3,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])

predBefore = model.predict(np.array([a, b, c]).transpose())

checkBefore = pd.DataFrame(data={
    "a" : a,
    "b" : b,
    "c" : c,
    "numpy" : ansNumpy,
    "element-wise" : ansElementWise,
    "before" : predBefore.tolist()
})

checkBefore.to_csv('checkBefore.csv')

history = model.fit(
    np.array([a, b, c]).transpose(),
    ansNumpy.transpose(),
    epochs=1000
)

predAfter = model.predict(np.array([a, b, c]).transpose())

checkAfter = pd.DataFrame(data={
    "a" : a,
    "b" : b,
    "c" : c,
    "numpy" : ansNumpy,
    "element-wise" : ansElementWise,
    "before" : predBefore.tolist(),
    "after" : predAfter.tolist()
})

checkAfter.to_csv('checkAfter.csv')