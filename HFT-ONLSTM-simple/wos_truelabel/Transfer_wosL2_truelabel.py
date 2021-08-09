# coding=utf-8
import os

from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pl
import numpy as np
import keras
from sklearn.metrics import classification_report, coverage_error, label_ranking_loss, \
    label_ranking_average_precision_score, dcg_score, ndcg_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from wos_textONlstmL2 import TextONLSTM2

maxlen = 502
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100

#  The path of embedding matrix
pretrained_w2v, _, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
#######################################################################################################################
print('Loading data...')
#  Path to the processed dataset
datafile = r'../wos_process/pretrained_wos.pkl'
with open(datafile, 'rb') as f:
    cont_pad, l1_1hot, l2_1hot, sem_l1 = pl.load(f)


pred_l1 = pl.load(open(r"../wos_process/wos_pred_l1", 'rb'))
pred_semantic_l1_pad = pl.load(open(r'../wos_process/wos_pred_semantic_l1_pad','rb'))
true_semantic_l1_pad = keras.preprocessing.sequence.pad_sequences(sem_l1, 2, padding='post', truncating='post')
x_train, x_test, y1_train, y1_test, y2_train, y2_test, true_semantic_l1_train, true_semantic_l1_test,\
    pred_l1_train, pred_l1_test= \
    train_test_split(cont_pad, l1_1hot, l2_1hot, true_semantic_l1_pad, pred_l1, test_size=0.2, random_state=42)


###################predicted label#####################################################################################
# 嵌入预测父标签
emb_label_train = list(np.column_stack((true_semantic_l1_train,x_train)))
emb_label_test = list(np.column_stack((true_semantic_l1_test,x_test)))
###############################################################################################1024####################
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../log/wos/wos_weight_l1.hdf5", by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
print("###############################################################################################################")

model.summary()
#######################################################################################################################
print('Train...')
fileweights = r"../log/wos/wos_weight_l2_with_true.hdf5"
if os.path.exists(fileweights):
    model.load_weights(fileweights)
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
# model.fit([emb_label_train], y2_train,
#           validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping, checkpoint, reduce_lr],
#
#           shuffle= True,
#           verbose= 2)
loss,accuracy = model.evaluate([emb_label_test], y2_test, verbose=0)
print('\ntest loss',loss,'accuracy',accuracy)

pred_l2_test =model.predict([emb_label_test], batch_size=batch_size)#此处传入的X_test和Input层的内容一致，要是多个Input()，就传个列表，和model.fit传入的参数一致(不传y_test)


one_hot_pred_l1 = pred_l1_test
one_hot_pred_l2 = pred_l2_test
one_hot_truth_l1 = y1_test
one_hot_truth_l2 = y2_test

pred_l2 = np.argmax(pred_l2_test, axis=1)
y2_test = np.argmax(y2_test, axis=1)


print(classification_report(y2_test, pred_l2, digits=8))

#####multilabel metrics in final layer######################

print("coverage_error at the highest levels:\n",coverage_error(one_hot_truth_l2, pred_l2_test))
print("label_ranking_loss at the highest levels:\n",label_ranking_loss(one_hot_truth_l2, pred_l2_test))
print("label_ranking_average_precision_score at the highest levels:\n", label_ranking_average_precision_score(one_hot_truth_l2, pred_l2_test))
print("dcg_score at the highest levels:\n",dcg_score(one_hot_truth_l2, pred_l2_test))
print("ndcg_score at the highest levels:\n",ndcg_score(one_hot_truth_l2, pred_l2_test))


#####multilabel metrics######################


pred = np.append(pred_l1_test, pred_l2_test, axis=1)

truth = np.append(one_hot_truth_l1, one_hot_truth_l2, axis=1)

print("coverage_error:\n",coverage_error(truth, pred))
print("label_ranking_loss:\n",label_ranking_loss(truth, pred))
print("label_ranking_average_precision_score:\n", label_ranking_average_precision_score(truth, pred))
print("dcg_score:\n",dcg_score(truth, pred))
print("ndcg_score:\n",ndcg_score(truth, pred))






