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
maxlen = 300
# maxlen = 310
# maxlen = 145
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
# print(pre_y1_pad[:3])
######
emb_pre_label_x = list(np.column_stack((pred_semantic_l1_pad,x)))
#真实标签310维
# emb_true_label_x = list(np.column_stack((y1_pad,x)))
########################################################################################################################
# x_train,x_test,y1_train,y1_test,y2_train,y2_test,pre_y1_train_pad,pre_y1_test_pad = train_test_split( x, y1, y2, pred_semantic_l1_pad, test_size=0.2, random_state=42)
# x_train,x_test,pre_y1_train_pad,pre_y1_test_pad=train_test_split( x, pred_semantic_l1_pad, test_size=0.2, random_state=42)

# x_train, x_test, y1_train, y1_test, y2_train, y2_test, pred_semantic_l1_train, pred_semantic_l1_test,\
#     pred_l1_train, pred_l1_test= \
#     train_test_split(x, y1, y2, pred_semantic_l1_pad, pred_l1, test_size=0.2, random_state=42)
# #预测标签
# emb_label_train = list(np.column_stack((pred_semantic_l1_train,x_train)))
# emb_label_test = list(np.column_stack((pred_semantic_l1_test,x_test)))

#真实标签
x_train, x_test, y1_train, y1_test, y2_train, y2_test, true_semantic_l1_train, true_semantic_l1_test,\
    pred_l1_train, pred_l1_test= \
    train_test_split(x, y1, y2, y1_pad[:,:3], pred_l1, test_size=0.2, random_state=42)
#预测标签
emb_label_train = list(np.column_stack((true_semantic_l1_train,x_train)))
emb_label_test = list(np.column_stack((true_semantic_l1_test,x_test)))

# x_train,x_test,y1_train_pad,y1_test_pad=train_test_split( x, y1_pad, test_size=0.2, random_state=42)
# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))
########################################################################################################################

print('Build model...')
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../dataset/dbp/output/dbp_l1_weights.hdf5",by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
##################################################################
model.summary()
print('Train...')
# fileweights = r"../dataset/dbp/output/dbp_l2_weights_with_true.hdf5"
fileweights = r"../log/dbp/dbp_l2_weights_without_l1.hdf5"
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
# checkpoint = ModelCheckpoint('weights.{epoch:03d}-{val_acc:.4f}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
##############################
# 当评价指标不在提升时，减少学习率

model.fit([x_train], y2_train,
          validation_split=0.1,
          verbose=2,
          batch_size=batch_size,
          epochs=epochs,
          # epochs = 1,
          callbacks=[early_stopping, checkpoint,reduce_lr],
          # validation_data=([emb_label_test], y2_test),
          shuffle= True)
#####################################
#####################################
print('category embedding')
# predict = model.predict([emb_pre_label_x])
# predict = np.argmax(predict, axis=1)
#
# pl.dump(predict, open('D:\E1106\pycharmModel0905\pycharmModel0905\htc-github\dbpedia\output\predictlabel\DBpedia_layer2_predict2_2', 'wb'))
loss, accuracy = model.evaluate([x_test],y2_test)
print('\ntest loss',loss,'accuracy',accuracy)

print("level2 without labels on the level1: end up")
