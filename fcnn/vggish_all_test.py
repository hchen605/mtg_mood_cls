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


from vggish.vggish import VGGish, VGGish_mtg
from vggish.preprocess_sound import *

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, classification_report, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0, help="data random seed")
#parser.add_argument("--eps", type=int, default=30, help="number of epochs")
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

x_test = np.load(DATA_ROOT + 'x_test.npy')


print('==== loading label ..,')

y_test = np.load(DATA_ROOT + 'y_test.npy')


print('=== processing mel spectrogram ====')
x_train, y_train = loading_data(x_test, y_test)

num_classes = 56



# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps
input_length = 160000

# Model

model = VGGish_mtg(num_classes)

weights_path = 'weight/weight_mtg_mood_seed{}_vggish/'.format(args.seed) + "best.hdf5"
model.load_weights(weights_path)

model.compile(loss='binary_crossentropy',
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
              metrics=[tensorflow.keras.metrics.BinaryCrossentropy()])

model.summary()


score = model.evaluate(x_test, y_test, verbose=0)
print('--- Test loss:', score[0])
print('- Test accuracy:', score[1])


#confusion matrix
y_pred = tensorflow.keras.activations.sigmoid(model.predict(x_test))
y_pred = model.predict(x_test)
#print('y_pred shape: ', y_pred.shape)

tags = pd.read_csv('../data/moodtheme_split.txt', delimiter='\t', header=None)[0].to_list()

thresholds, decisions = calculate_decisions(y_test, y_pred, tags, './threshold/threshold_vggish.csv', './threshold/decision_vggish', display=0)

ap = average_precision_score(y_test, y_pred)
print('PR AUC: ', ap)


roc = roc_auc_score(y_test, y_pred)
print('ROC AUC: ', roc)





'''
cm = confusion_matrix(y_test_, y_pred_)
#print(cm)

classes_12 = ['C1','C2','C3','C4','D1','D2','D3','D4','D5','M1','M3','P1','P2','P3','P4','P5','P6']
#disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_12)
disp.plot(cmap=plt.cm.Blues)
plt.title('Microphone Classification')
plt.show()
#plt.savefig('./music_log/cm_{}_{}.pdf'.format(mic,target))
plt.savefig('./confusion/cm_full_mobile_class_{}_limit_{}_seed_{}_unseen_small_1m.pdf'.format(args.nclass, args.limit, args.seed))
'''