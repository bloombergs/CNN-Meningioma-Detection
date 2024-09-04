import matplotlib.pyplot
import tensorflow
import numpy

preprocess = tensorflow.keras.preprocessing.image.ImageDataGenerator(rescale =1/255)

testflow = preprocess.flow_from_directory(r"C:\Users\Hello\Desktop\cnnbraintumordetection\test",
                                       target_size=(200,200),
                                       batch_size=32,
                                       class_mode = "categorical")

validationflow = preprocess.flow_from_directory(r"C:\Users\Hello\Desktop\cnnbraintumordetection\validation",
                                       target_size=(200,200),
                                       batch_size=32,
                                       class_mode = "categorical")

trainflow = preprocess.flow_from_directory(r"C:\Users\Hello\Desktop\cnnbraintumordetection\train",
                                       target_size=(200,200),
                                       batch_size=32,
                                       class_mode = "categorical")

model = tensorflow.keras.models.Sequential([
     tensorflow.keras.layers.Conv2D(16,(2,2),input_shape=[200,200,3],activation="relu"),
     tensorflow.keras.layers.MaxPooling2D(2,2),
     tensorflow.keras.layers.Conv2D(16,(2,2),activation="relu"),
     tensorflow.keras.layers.MaxPooling2D(2,2),
     tensorflow.keras.layers.Conv2D(16,(2,2),activation="relu"),
     tensorflow.keras.layers.MaxPooling2D(2,2),
     tensorflow.keras.layers.Flatten(),
     tensorflow.keras.layers.Dense(512,activation="relu"),
     tensorflow.keras.layers.Dense(2,activation="softmax")
 ])
print(testflow.batch_size)
print(testflow.class_indices)
model.compile(optimizer="adam",loss="categorical_crossentropy")

model.fit(trainflow,epochs=10,validation_data=validationflow)

test_images, test_labels = next(iter(testflow))

img = test_images[3]
label = test_labels[3]

prediction = model.predict(numpy.expand_dims(img, axis=0))

predicted_class = numpy.argmax(prediction, axis=1)[0]

actual_class = numpy.argmax(label, axis=0)

class_names = ['Class 0', 'Class 1']  

print(f"Prediction: {class_names[predicted_class]}")
print(f"Actual Label: {class_names[actual_class]}")

matplotlib.pyplot.imshow(img)
matplotlib.pyplot.show()

print(trainflow.class_indices)
model.save("meningiomapred.h5")