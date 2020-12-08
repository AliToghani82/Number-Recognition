import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np



mnist = tf.keras.datasets.mnist
(x_train, y_train),(x_test, y_test) = mnist.load_data()
plt.imshow(x_train[6], cmap="gray")

x_train = tf.keras.utils.normalize(x_train, axis=1)

x_test = tf.keras.utils.normalize(x_test, axis=1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(x=x_train, y=y_train, epochs=10)

test_loss, test_acc = model.evaluate(x=x_test, y=y_test)

predictions = model.predict([x_test])

print(np.argmax(predictions[120]))

plt.imshow(x_test[120], cmap="gray")
plt.show()
