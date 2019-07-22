import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images / 255.0
test_images = test_images / 255.0

model = keras.Sequential()
model.add(keras.layers.Flatten(input_shape=(28, 28)))
model.add(keras.layers.Dense(128, activation=tf.nn.leaky_relu))
model.add(keras.layers.Dense(10, activation=tf.nn.softmax))

model.compile(loss=keras.losses.sparse_categorical_crossentropy, metrics=['accuracy'],
              optimizer=keras.optimizers.Adam(lr=1e-3))

history = model.fit(train_images, train_labels, epochs=10)

test_loss, test_accuracy = model.evaluate(test_images, test_labels)

print('loss: ', test_loss)
print('accuracy: ', test_accuracy)

predictions = model.predict(test_images)

plt.figure(figsize=(10, 10))

for i in range(15):
    plt.subplot(5, 6, 2 * i + 1)
    plt.imshow(test_images[i])
    predicted_label = np.argmax(predictions[i])
    true_label = test_labels[i]

    if predicted_label == true_label:
        plt.xlabel("{}".format(class_names[predicted_label]), color='blue')
    else:
        plt.xlabel("{} ({})".format(class_names[predicted_label], class_names[test_labels[i]]), color='red')

    plt.subplot(5, 6, 2 * i + 2)
    plt.bar(range(10), predictions[i])

plt.show()
