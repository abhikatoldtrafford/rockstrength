import keras,cv2, random
import numpy as np
import pandas as pd
from keras import metrics
from keras.layers.core import Dense, Activation
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Model, load_model, Sequential
from keras.applications import imagenet_utils
from keras.layers import Dense,GlobalAveragePooling2D, Flatten, Dropout
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.optimizers import Adam
from sklearn.utils import class_weight
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback
import tensorflow as tf
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
if tf.test.gpu_device_name():
    print('GPU found')
else:
    print("No GPU found")

traindf=pd.read_csv('./rock_with_ucs/train_ucs_aug.csv',names=['path', 'ucs', 'range'], header= None, dtype=str)

train_data_dir = './rock_with_ucs'
batch_size = 8
img_width, img_height = 512, 512



def get_random_eraser(p=0.25, s_l=0.02, s_h=0.25, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=False):
    def eraser(input_img):
        img_h, img_w, img_c = input_img.shape
        p_1 = np.random.rand()
        if p_1 > p:
            return preprocess_input(input_img)
        while True:
            s = np.random.uniform(s_l, s_h) * img_h * img_w
            r = np.random.uniform(r_1, r_2)
            w = int(np.sqrt(s / r))
            h = int(np.sqrt(s * r))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            if left + w <= img_w and top + h <= img_h:
                break

        if pixel_level:
            c = np.random.uniform(v_l, v_h, (h, w, img_c))
        else:
            c = np.random.uniform(v_l, v_h)
        input_img[top:top + h, left:left + w, :] = c
        return preprocess_input(input_img)
    return eraser

train_datagen = ImageDataGenerator(
    zoom_range=0.3,
    shear_range=0.2,
    channel_shift_range=100.0,
    horizontal_flip=True,
    rotation_range=30,
    vertical_flip=True,
    validation_split=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=(0.1, 0.9),
    preprocessing_function = (get_random_eraser(pixel_level=True, p = .5)))  # set validation split


test_dataget = ImageDataGenerator(validation_split = .2, preprocessing_function = preprocess_input)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=traindf,
    directory='.',
    x_col = 'path',
    y_col = 'range',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training') # set as training data


validation_generator = test_dataget.flow_from_dataframe(
    dataframe=traindf,
    directory='.',
    x_col = 'path',
    y_col = 'range',
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle = True,
    subset='validation') # set as validation data


unique, counts = np.unique(train_generator.classes, return_counts=True)
class_weights = class_weight.compute_class_weight(
           'balanced',
            np.unique(train_generator.classes),
            train_generator.classes)
listy = list(class_weights)
class_weights_dict = (dict(enumerate(listy)))
label_map = (train_generator.class_indices)
print(label_map)
print(class_weights_dict)

model1 = load_model('model_ucs_latest.h5')
model = Sequential()

for layer in model1.layers[:-3]:
    layer.trainable = True
    model.add(layer)
for layer in model1.layers[-3:]:
    layer.trainable = True
    model.add(layer)

#model.add(Dense(5,activation='softmax', name = 'classify'))
#model.add(Dense(1,activation='relu', name = 'ucs_value'))

adm=Adam(lr = 1e-5)

model.compile(optimizer=adm,loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
mcp_save = ModelCheckpoint('model_ucs_latest.h5', save_best_only=True, monitor='val_loss', mode='min', verbose = 1)
model.fit_generator(
    train_generator,
    class_weight = class_weights_dict,
    steps_per_epoch = train_generator.samples // (batch_size),
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // (batch_size),
    epochs = 1000,
    verbose = 1,
    callbacks=[mcp_save])

