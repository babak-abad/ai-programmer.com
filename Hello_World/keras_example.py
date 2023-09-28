from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, ReLU, Softmax
import cv2.load_config_py3
import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.callbacks import ModelCheckpoint
from tensorflow.python.keras import backend as K

HEIGHT = 32
WIDTH = 32
DEPTH = 3
VLD_SZ = 0.3
DS_DIR = '/home/babak/Downloads/dataset/r7bthvstxw-1'
INIT_LR = 0.01
EPOCHS = 100
BATCH_SZ = 32

def make_mdl(num_class, input_w, input_h, input_d):
    inp_shape = (input_h, input_w, input_d)

    if K.image_data_format() == 'channels_first':
        inp_shape = (input_d, input_h, input_w)

    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), input_shape=inp_shape))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(32, kernel_size=(3, 3)))
    model.add(ReLU())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(num_class))
    model.add(Softmax())
    return model

# loading images
im_paths = glob.glob(os.path.join(DS_DIR, '*/*.jpg'))

x = []
y = []

# reading x and y
for p in im_paths:
    y.append(p.split(sep=os.path.sep)[-2])
    if DEPTH == 1:
        im = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
    else:
        im = cv2.imread(p, cv2.IMREAD_COLOR)
    im = cv2.resize(src=im, dsize=(WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC)
    im = x.append(im)

enc = LabelEncoder()
# normalizing x and y
x = np.array(x, dtype='float')/255.0
y = enc.fit_transform(y)
y = tf.keras.utils.to_categorical(y)

#spliting data
(x_trn, x_vld, y_trn, y_vld) = train_test_split(x, y, test_size=VLD_SZ, random_state=42)

#determining num_class by number of folders in dataset
num_class = -1 # because os.walk return the input folder as result
for p in os.walk(DS_DIR):
    num_class+=1

mdl = make_mdl(num_class=num_class, input_w=WIDTH, input_h=HEIGHT, input_d=DEPTH)
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
mdl.compile(optimizer=opt, loss=tf.keras.losses.categorical_crossentropy, metrics=['accuracy'])

fn = os.path.sep.join(['check_points',
	"weights-{epoch:03d}-{val_loss:.4f}.hdf5"])

chk_pnt = ModelCheckpoint(
    fn,
    monitor="val_loss",
    mode="min",
	save_best_only=True,
    verbose=1)

mdl.fit(
    x_trn, y_trn,
    batch_size = BATCH_SZ,
    epochs=EPOCHS,
    validation_data=(x_vld, y_vld),
    steps_per_epoch=len(x_trn)//BATCH_SZ,
    callbacks=[chk_pnt])