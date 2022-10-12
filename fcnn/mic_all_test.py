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
from models.small_fcnn_att import model_fcnn, model_fcnn_pre
from models.attRNN import AttRNN
#from models.xvector import model_xvector
#from models.attRNN import AttRNN, AttRNN_pre

from tensorflow.compat.v1 import ConfigProto, InteractiveSession


from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import precision_recall_curve, auc, confusion_matrix, f1_score, classification_report, roc_auc_score, average_precision_score

parser = argparse.ArgumentParser()
parser.add_argument("--gender", type=int, default=0, help="full (0), female(1), male (2)")
parser.add_argument("--nclass", type=int, default=0, help="3class (0), 12class(1)")
parser.add_argument("--limit", type=int, default=100, help="number of data")
parser.add_argument("--seed", type=int, default=0, help="data random seed")
parser.add_argument("--eps", type=int, default=30, help="number of epochs")
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


num_classes = 56



# +
# Parameters
num_freq_bin = 128
num_audio_channels = 1
batch_size = 32
epochs = args.eps
input_length = 160000

# Model
model = model_fcnn(num_classes, input_shape=[num_freq_bin, None, num_audio_channels], num_filters=[24, 48, 96], wd=0)
#model = model_xvector(num_classes)
#model = AttRNN(num_classes, input_length)
#model = AttRNN_pre(num_classes, input_length)

weights_path = 'weight/weight_mtg_mood_seed{}_attcnn_pos/'.format(args.seed) + "best.hdf5"
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
#print('y_pred shape: ', y_pred.shape)

tags = pd.read_csv('../data/moodtheme_split.txt', delimiter='\t', header=None)[0].to_list()

thresholds, decisions = calculate_decisions(y_test, y_pred, tags, './threshold/threshold_attcnn_pos.csv', './threshold/decision_attcnn_pos', display=0)

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