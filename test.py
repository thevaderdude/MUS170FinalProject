import tensorflow as tf
from tensorflow.keras import layers, models
import joblib
import numpy as np


x, y = joblib.load('train_data.joblib')
x = x.reshape(x.shape[0], x.shape[1], x.shape[2], 1)

p = np.random.permutation(len(x))
x, y = x[p], y[p]

# fix train labels
num_labels = y.shape[1]
new_y = np.zeros(y.shape[0])
for i in range(len(new_y)):
    for j in range(y.shape[1]):
        if y[i, j] == 1:
            new_y[i] = j
y = new_y

# split into train and test
test_partition = len(x) - 10000
x_train = x[:test_partition]
x_test = x[test_partition:]
y_train = y[:test_partition]
y_test = y[test_partition:]

print(x.shape[1:])

model = models.Sequential()
model.add(layers.Conv2D(32, (2, 2), activation='relu', input_shape=x.shape[1:]))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (2, 2), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
print(model.summary())
model.add(layers.Conv2D(64, (2, 2), activation='relu'))

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(num_labels))

print(model.summary())
model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

model.save('saved_model/my_model')

history = model.fit(x_train, y_train, epochs=20, validation_data=(x_test, y_test))

model.save('saved_model/my_model')
