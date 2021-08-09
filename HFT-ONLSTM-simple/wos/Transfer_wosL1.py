# coding=utf-8
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
import pickle as pl
import keras
from wos_textONlstmL1 import TextONLSTM
from dbp_process.probabilityonehot import trans_to_onehot

maxlen = 500
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

#  The path of embedding matrix
pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
########################################################################################################################
print('Loading data...')

#  Path to the processed dataset


datafile = r'../wos_process/pretrained_wos.pkl'
with open(datafile, 'rb') as f:
    cont_pad, l1_1hot, l2_1hot, sem_l1 = pl.load(f)
x_train, x_test, y1_train, y1_test = train_test_split(cont_pad, l1_1hot, test_size=0.2, random_state=42)
########################################################################################################################
print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()

########################################################################################################################
print('Train...')

# The path to save the weight of the first level training
fileweights = r"../log/wos/wos_weight_l1.hdf5"
if os.path.exists(fileweights):
    model.load_weights(fileweights)

checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
# model.fit(x_train, y1_train,
#           validation_split=0.1,
#           batch_size=batch_size,
#           # epochs=epochs,
#           verbose = 2,
#           epochs = 100,
#           callbacks=[early_stopping, checkpoint,reduce_lr],
#           # validation_data=(x_test, y1_test),·
#           shuffle = True
#           )

########################################################################################################################
lossl1, accl1 = model.evaluate(x_test, y1_test, verbose=0)
print('lossl1', lossl1, ' accl1', accl1)
print('category Embedding')
pred_l1 = model.predict([cont_pad])

pl.dump(pred_l1, open(r'../wos_process/wos_pred_l1', 'wb'))
pred_l1 = np.argmax(pred_l1, axis=1)

# The path of the ID set of the predicted first level label


semantic_l1 = ['Computer Science', 'Electrical Engineering', 'Psychology', 'Mechanical Engineering', 'Civil Engineering',
      'Medical Sciences', 'Biochemistry']
pred_semantic_l1 = []

for i in pred_l1:
    pred_semantic_l1.append([word_to_id[x] for x in semantic_l1[i].split(' ') if x in word_to_id])
l1_length = 2
pred_semantic_l1_pad = keras.preprocessing.sequence.pad_sequences(pred_semantic_l1, l1_length, padding='post', truncating='post')

#  The path of the set of semantic vectors embedded in matrix mapping for label ID
with open(r'../wos_process/wos_pred_semantic_l1_pad', 'wb') as f:
    pl.dump(pred_semantic_l1_pad, f)
