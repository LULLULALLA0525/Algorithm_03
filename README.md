## Algorithm_03

20183622 김태우 Algorithm Assignment #5

---
## TensorFlow_mnist_example(1)
### MODEL1: 3 Layers with 1 Convolution layer
```python
def select_model(model_number):
    if model_number == 1:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2 
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 3
```

```python
model = select_model(1)
```

### Training Step: Training for 5 epochs.

```python
model.fit(train_images, train_labels,  epochs = 5)
```

```
Epoch 1/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.7381 - accuracy: 0.9374
Epoch 2/5
1875/1875 [==============================] - 13s 7ms/step - loss: 0.0874 - accuracy: 0.9749
Epoch 3/5
1875/1875 [==============================] - 12s 7ms/step - loss: 0.0714 - accuracy: 0.9787
Epoch 4/5
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0606 - accuracy: 0.9810
Epoch 5/5
1875/1875 [==============================] - 12s 6ms/step - loss: 0.0563 - accuracy: 0.9830
<keras.callbacks.History at 0x2a11ecc9f70>
```

### Test Step
### Perform Test with Test data

```python
test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)
print('Test accuracy :', accuracy)
```

```
313/313 - 1s - loss: 0.1057 - accuracy: 0.9760 - 843ms/epoch - 3ms/step

Test loss :  0.10574308782815933
Test accuracy : 0.9760000109672546
```

### Before prediction, change test image's type to float 32.

```python

test_images = tf.cast(test_images, tf.float32)
pred = model.predict(test_images)
Number = [0,1,2,3,4,5,6,7,8,9]
```

```
313/313 [==============================] - 1s 2ms/step
```

```python
print('Prediction : ', pred.shape)
print('Test labels : ', test_labels.shape)
```

```
Prediction :  (10000, 10)
Test labels :  (10000,)
```

### Functions for plot images, probability

```python
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(Number[predicted_label],
                                100*np.max(predictions_array),
                                Number[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  plt.xticks(Number)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  
  i = 1
  plt.figure(figsize=(6,3))
  plt.subplot(1,2,1)
  plot_image(i, pred, test_labels, test_images)
  plt.subplot(1,2,2)
  plot_value_array(i, pred,  test_labels)
  plt.show()
```


![plot error (1)](https://user-images.githubusercontent.com/69046742/173228049-60957996-36e9-46a4-ab07-6fca7cf23b02.png)
### TensorFlow_mnist_example(2)'s plot_error
![plot error (2)](https://user-images.githubusercontent.com/69046742/173228051-294a769e-75cf-4734-a740-bb08ad07bbc3.png)
### TensorFlow_mnist_example(3)'s plot_error
![plot error (3)](https://user-images.githubusercontent.com/69046742/173228052-8300ab57-829f-47e0-8c39-4ae9daa01bcd.png)
