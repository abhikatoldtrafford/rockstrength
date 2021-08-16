from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from keras.applications.mobilenet_v2 import preprocess_input
import glob
from tqdm import tqdm
from time import time
img_width, img_height = 512, 512

# load the model we saved
model = load_model('model_ucs_regression.h5')
#model.compile(loss='binary_crossentropy',optimizer='rmsprop',metrics=['accuracy'])

# predicting images
files = glob.glob('rock_with_ucs\*\*.jpg')
listy = []
for f in tqdm(files):
    start = time()
    img = image.load_img(f, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = preprocess_input(np.expand_dims(x, axis=0))

    images = np.vstack([x])
    pred = model.predict(images)
    end = time()
    listy.append([end-start])
import pandas as pd
df = pd.DataFrame(listy)
df.to_csv('time.csv', index=False)