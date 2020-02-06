import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model


source = "./data/image_dataset"


def load_data():

    for item in os.walk(source):
        class_dir = item[1]
        break

    print(class_dir)
    n_classes = len(class_dir)
    print("Number of Classes recognized : ", n_classes)
    return class_dir


class_dir = load_data()
n_classes = len(class_dir)


def get_n_samples(class_dir):

    for item in class_dir:
        for i in os.walk(source+'/'+str(item)):
            print(len(i[2]))


n_samples = 340
x_train = np.zeros((n_samples*n_classes, 2700))
x_train_img = np.zeros( (n_samples*n_classes, 30, 30, 3) )
y_train = np.zeros((n_samples*n_classes))
y_train_img = np.zeros( (n_samples*n_classes))


for item_no in range(n_classes):
    for i in range(n_samples):
        try:
            # print(source+'/'+str(class_dir[0])+"/"+str(i))
            img = cv2.imread(source+'/'+str(class_dir[item_no])+'/'+str(i)+'.png')
        except:
            print("Read Error")

        img = np.asarray(img)
        x_train_img[ (item_no*n_samples)+i,:] = img
        img = img.flatten()
        x_train[ (item_no*n_samples)+i, :] = img
        y_train[ (item_no*n_samples)+i]= item_no
        y_train_img = y_train

#print(x_train)
#print(y_train)

train_ds = tf.data.Dataset.from_tensor_slices(
    (x_train_img, y_train)).shuffle(100).batch(8)
test_ds = tf.data.Dataset.from_tensor_slices((x_train_img, y_train)).batch(32)



class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(16, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(64, activation='relu')
        self.d2 = Dense(n_classes, activation='softmax')

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

model = MyModel()

loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

print("Test passed")

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
    predictions = model(images)
    t_loss = loss_object(labels, predictions)

    test_loss(t_loss)
    test_accuracy(labels, predictions)

EPOCHS = 5

for epoch in range(EPOCHS):
    for images, labels in train_ds:
        train_step(images, labels)

    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))

  # Reset the metrics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()



#y_pred = model.predict(train_ds)
#matrix = metrics.confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

print("Done")

