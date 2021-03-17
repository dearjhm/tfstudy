import tensorflow as tf
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

xs = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
ys = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=float)

model = Sequential(Dense(units=1, input_shape=[1]))
model.compile(loss="mean_squared_error", optimizer="sgd", metrics=["accuracy"])
model.fit(xs, ys, epochs=1000)

y_pred = model.predict([7.0, 8.0, 9.0, 10.0, 11.0])
print(y_pred)
