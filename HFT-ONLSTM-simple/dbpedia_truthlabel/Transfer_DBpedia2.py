# coding=utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import time
import numpy as np
from sklearn.model_selection import train_test_split
from dbpedia_textONlstmL2 import TextONLSTM2
import pickle as pl
from dbp_process.probabilityonehot import trans_to_onehot
import keras
maxlen = 303

max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

start_time =time.time()
#  The path of embedding matrix
pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')
#  Path to the processed dataset
x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'../dataset/dbp/DBP_txt_vector300dim_y1y2y3_10dim_zjp', 'rb'))

pred_l1 = pl.load(open(r"../dataset/dbp/predictlabel/dbp_pred_l1", 'rb'))
pred_semantic_l1_pad = pl.load(open(r'../dataset/dbp/predictlabel/dbp_pred_semantic_l1_pad','rb'))

######
emb_pre_label_x = list(np.column_stack((pred_semantic_l1_pad,x)))


#真实标签
x_train, x_test, y1_train, y1_test, y2_train, y2_test, true_semantic_l1_train, true_semantic_l1_test,\
    pred_l1_train, pred_l1_test= \
    train_test_split(x, y1, y2, y1_pad[:,:3], pred_l1, test_size=0.2, random_state=42)
#预测标签
emb_label_train = list(np.column_stack((true_semantic_l1_train,x_train)))
emb_label_test = list(np.column_stack((true_semantic_l1_test,x_test)))


########################################################################################################################

print('Build model...')
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../dataset/dbp/output/dbp_l1_weights.hdf5",by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
##################################################################
model.summary()
print('Train...')

fileweights = r"../log/dbp/dbp_l2_weights_with_true.hdf5"
if os.path.exists(fileweights):
    model.load_weights(fileweights)
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
##############################
# 当评价指标不在提升时，减少学习率

# model.fit([emb_label_train], y2_train,
#           validation_split=0.1,
#           verbose=2,
#           batch_size=batch_size,
#           epochs=epochs,
#
#           callbacks=[early_stopping, checkpoint,reduce_lr],
#
#           shuffle= True)
#####################################
#####################################
print('category embedding')
loss, accuracy = model.evaluate([emb_label_test],y2_test, verbose=0)
print('loss',  loss, 'accuracy', accuracy )
# pred_l2 = model.predict([emb_label_test])

print("level2: end up")
