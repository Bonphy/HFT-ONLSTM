# coding=utf-8
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
x,y1,y2,y1_pad,y2_pad = pl.load(open(r'../dataset/wos/WOSDATA_txt_vector500dimsy1y2_10dim_zjp','rb'))
x_train, x_test, y1_train, y1_test = train_test_split( x, y1, test_size=0.2, random_state=42)
########################################################################################################################
print('Build model...')
model = TextONLSTM(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
# fileweights = r"../dataset/wos/output/Ay1pad_y2_best_weights.h5"
# model.load_weights(fileweights)
# model.evaluate(x_test, y1_test)
# model.predict()
########################################################################################################################
print('Train...')

# The path to save the weight of the first level training
fileweights = r"../log/wos/wos_weight_l1.hdf5"


checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
model.fit(x_train, y1_train,
          validation_split=0.1,
          batch_size=batch_size,
          # epochs=epochs,
          verbose=2,
          epochs = 100,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          # validation_data=(x_test, y1_test),
          shuffle= True
          )
# model.save( r"../dataset/wos/output/l1_model.h5")
########################################################################################################################
print('category Embedding')
pred_l1 = model.predict([x])

pl.dump(trans_to_onehot(pred_l1), open(r'../dataset/wos/predictlabel/wos_pred_l1', 'wb'))
pred_l1 = np.argmax(pred_l1, axis=1)

# The path of the ID set of the predicted first level label

# pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings\emb_matrix_glove_300', 'rb'))
semantic_l1 = ['biochemistry', 'civil', 'computer science', 'electrical', 'mechanical', 'medical', 'psychology']
pred_semantic_l1 = []

for i in pred_l1:
    pred_semantic_l1.append([word_to_id[x] for x in semantic_l1[i].split(' ') if x in word_to_id])
l1_length = 2
pred_semantic_l1_pad = keras.preprocessing.sequence.pad_sequences(pred_semantic_l1, l1_length, padding='post', truncating='post')

#  The path of the set of semantic vectors embedded in matrix mapping for label ID
with open(r'../dataset\wos\predictlabel\wos_pred_semantic_l1_pad', 'wb') as f:
    pl.dump(pred_semantic_l1_pad, f)
