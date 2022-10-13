import os
import sys
sys.path.append("..")
import argparse

import tensorflow
import numpy as np
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam

from utils import *
from funcs import *

from ts_dataloader import *

from vggish.vggish import VGGish, VGGish_mtg
from vggish.preprocess_sound import *


from tensorflow.compat.v1 import ConfigProto, InteractiveSession

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=100, help="number of epochs")
args = parser.parse_args()


#tensorflow.reset_default_graph()
os.environ['PYTHONHASHSEED']= str(args.seed)
tensorflow.random.set_seed(args.seed)
tensorflow.compat.v1.set_random_seed(args.seed)
random.seed(args.seed)
#tensorflow.keras.utils.set_random_seed(1)
np.random.seed(args.seed)


config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)



DATA_ROOT = '../data/'
print('==== loading  data ====')
#train = load_data(train_csv)
#dev = load_data(dev_csv)
#test = load_data(test_csv)
x_train = np.load(DATA_ROOT + 'x_train.npy')
x_dev = np.load(DATA_ROOT + 'x_dev.npy')

print('==== loading label ..,')

y_train = np.load(DATA_ROOT + 'y_train.npy')
y_dev = np.load(DATA_ROOT + 'y_dev.npy')

print('=== processing mel spectrogram ====')
x_train, y_train = loading_data(x_train, y_train)
x_dev, y_dev = loading_data(x_dev, y_dev)

#print ("=== Number of training data: {}".format(len(train)))
#print ("=== Number of test data: {}".format(len(test)))

num_classes = 56

# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps
input_length = 160000

# Model
model = VGGish_mtg(num_classes)
#weights_path = '/home/hsinhung/mic_acoustic/12cla ss/fcnn/weight/weight_full_mobile_limit400_seed0_audioset/full/3class/best.hdf5'
#model.load_weights(weights_path)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=[tensorflow.keras.metrics.BinaryCrossentropy()])
'''
def my_loss(pos_weight):
    def weighted_cross_entropy_with_logits(labels, logits):
        loss = tensorflow.nn.weighted_cross_entropy_with_logits(
            labels, logits, pos_weight
        )
        return loss
    return weighted_cross_entropy_with_logits
pos_weight = np.array([33.61148649,37.68560606,27.4077135,24.62623762,51.5492228, 37.26217228, 65.45394737,  27.03532609,  52.08900524,  58.52352941, 28.50716332,  16.04677419,  23.19114219,  29.3480826,   39.32411067, 51.02051282,  19.97791165,  19.93787575,  11.60910152,  16.55407654, 160.46774194,  13.33646113,  45.22272727,  91.27522936,  58.86982249, 127.55128205,  10.73247033, 101.52040816, 101.52040816,  72.62043796, 17.73440285,  20.18052738,  26.60160428,  43.63596491,  16.63712375, 26.74462366,  48.06280193,  72.62043796,  78.33858268,  26.46010638, 93.85849057,  13.03931848,  71.57553957,  27.4077135,   26.53066667, 168.62711864,  33.72542373,  47.83173077,  23.97349398,  27.48342541, 49.49751244,  38.11877395,  62.57232704, 111.78651685, 46.06018519, 22.15812918])
#loss_fn = tensorflow.nn.weighted_cross_entropy_with_logits(pos_weight=pos_weight)
model.compile(loss=my_loss(pos_weight),
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=[my_loss(pos_weight)])
'''
model.summary()

# Checkpoints
if not os.path.exists('weight/weight_mtg_mood_seed{}_vggish/'.format(args.seed)):
    os.makedirs('weight/weight_mtg_mood_seed{}_vggish/'.format(args.seed))

#save_path = "weight/weight_full_mobile_limit{}_seed{}/".format(args.limit, args.seed)+ experiments + "{epoch:02d}-{val_accuracy:.4f}.hdf5"
save_path = "weight/weight_mtg_mood_seed{}_vggish/".format(args.seed) + "best.hdf5"
checkpoint = keras.callbacks.ModelCheckpoint(save_path, monitor='val_loss', verbose=1, save_best_only=True)
callbacks = [checkpoint]


# Training
exp_history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1,
              validation_data=(x_dev, y_dev), callbacks=callbacks)

print("=== Best Val. Loss: ", min(exp_history.history['val_loss']), " At Epoch of ", np.argmin(exp_history.history['val_loss'])+1)

# +

