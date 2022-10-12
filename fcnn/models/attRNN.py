
from tensorflow.keras.models import Model, load_model

from tensorflow.keras import layers as L
from tensorflow.keras import backend as K
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras import models

from kapre.time_frequency import Melspectrogram, Spectrogram
from kapre.utils import Normalization2D



def AttRNN(nCategories, inputLength=160000):
    # simple LSTM
    rnn_func=L.LSTM
    sr = 16000
    iLen = inputLength
   

    inputs = L.Input((inputLength,), name='input')

    x = L.Reshape((1, -1))(inputs)

    m = Melspectrogram(n_dft=1024, n_hop=128, input_shape=(1, iLen),
                       padding='same', sr=sr, n_mels=80,
                       fmin=40.0, fmax=sr / 2, power_melgram=1.0,
                       return_decibel_melgram=True, trainable_fb=False,
                       trainable_kernel=False,
                       name='mel_stft')
    m.trainable = False

    x = m(x)

    x = Normalization2D(int_axis=0, name='mel_stft_norm')(x)

    # note that Melspectrogram puts the sequence in shape (batch_size, melDim, timeSteps, 1)
    # we would rather have it the other way around for LSTMs

    x = L.Permute((2, 1, 3))(x)

    x = L.Conv2D(10, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)
    x = L.Conv2D(1, (5, 1), activation='relu', padding='same')(x)
    x = L.BatchNormalization()(x)

    # x = Reshape((125, 80)) (x)
    # keras.backend.squeeze(x, axis)
    x = L.Lambda(lambda q: K.squeeze(q, -1), name='squeeze_last_dim')(x)

    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]
    x = L.Bidirectional(rnn_func(64, return_sequences=True)
                        )(x)  # [b_s, seq_len, vec_dim]

    xFirst = L.Lambda(lambda q: q[:, -1])(x)  # [b_s, vec_dim]
    query = L.Dense(128)(xFirst)

    # dot product attention
    attScores = L.Dot(axes=[1, 2])([query, x])
    attScores = L.Softmax(name='attSoftmax')(attScores)  # [b_s, seq_len]

    # rescale sequence
    attVector = L.Dot(axes=[1, 1])([attScores, x])  # [b_s, vec_dim]

    #x = L.Dense(64, activation='relu')(attVector)
    x = L.Dense(128, activation='relu')(attVector)
    #x = L.Dense(32)(x)
    x = L.Dense(64)(x)

    output = L.Dense(nCategories, activation='softmax', name='output')(x)

    model = Model(inputs=[inputs], outputs=[output])

    return model

def AttRNN_pre(num_classes, inputLength):
    attrnn = AttRNN(nCategories=36, inputLength=inputLength)
    print('loading GSC pre-trained weight')
    weights_path = '/home/hsinhung/SpeechCmdRecognition/model-attRNN.h5'
    attrnn.load_weights(weights_path)
    #attrnn = models.load_model(weights_path, custom_objects={'Melspectrogram': Melspectrogram, 'Normalization2D': Normalization2D })
    #model.add(Dense(num_classes, input_shape=(12,)))
    model = models.Sequential()
    model.add(attrnn)
    model.add(L.Dense(units=num_classes, activation='softmax'))
    #attrnn.trainable = False
    return model