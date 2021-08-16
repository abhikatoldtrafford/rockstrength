from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
import glob
# dimensions of our images
img_width, img_height = 512, 512

# load the model we saved
model = load_model('model_ucs_latest.h5')
#model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# predicting images
files = glob.glob('validation\*\*.*')
#print(files)
for f in files:
    img = image.load_img(f, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))

    images = np.vstack([x])
    pred = np.argmax(model.predict(images), axis=-1)

    if pred == [0]:
        print(f,'0to15')
    elif pred ==[1]:
        print(f,'16to30')
    elif pred ==[2]:
        print(f,'31to50')
    elif pred ==[3]:
        print(f,'50to80')
    elif pred ==[4]:
        print(f,'80to100')
# # predicting multiple images at once
# img = image.load_img('test2.jpg', target_size=(img_width, img_height))
# y = image.img_to_array(img)
# y = np.expand_dims(y, axis=0)
#
# # pass the list of multiple images np.vstack()
# images = np.vstack([x, y])
# classes = model.predict_classes(images, batch_size=10)
#
# # print the classes, the images belong to
# print classes
# print classes[0]
# print classes[0][0]