import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense


faces_data = np.load('faces_data.npy')
labels = np.load('labels.npy')


faces_data = faces_data.astype('float32') / 255.0


train_images = faces_data[:800]
train_labels = labels[:800]
test_images = faces_data[800:]
test_labels = labels[800:]


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model.fit(train_images, train_labels, epochs=20, batch_size=20, validation_data=(test_images, test_labels))


test_loss, test_acc = model.evaluate(test_images, test_labels)


model.save('faces_detection_model.h5')


model = load_model('faces_detection_model.h5')


camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    frame = cv2.resize(frame, (150, 150))
    faces = np.expand_dims(frame, axis=0)
    faces = faces.astype('float32') / 255.0
    prediction = model.predict(faces)[0]
    if prediction > 0.5:
        print('Face detected')
    else:
        print('No face detected')

    cv2.imshow('Camera', frame)

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()
