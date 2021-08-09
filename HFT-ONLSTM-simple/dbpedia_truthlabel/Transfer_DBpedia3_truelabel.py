# coding=utf-8
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pickle as pl
import numpy as np
from sklearn.model_selection import train_test_split

from dbpedia_textONlstmL3 import TextONLSTM3
from dbp_process.probabilityonehot import trans_to_onehot
from sklearn.metrics import classification_report, coverage_error, label_ranking_loss, \
    label_ranking_average_precision_score, dcg_score, ndcg_score, hamming_loss

maxlen = 303
# maxlen = 303
# maxlen = 145
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100
#  The path of embedding matrix
pretrained_w2v, word_to_id, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
################################################################
print('Loading data...')
#  Path to the processed dataset
x,y1,y2,y3,y1_pad,y2_pad,y3_pad = pl.load(open(r'../dataset/dbp/DBP_txt_vector300dim_y1y2y3_10dim_zjp', 'rb'))

pred_l1 = pl.load(open(r"../dataset/dbp/predictlabel/dbp_pred_l1", 'rb'))
pred_l2 = pl.load(open(r"../dataset/dbp/predictlabel/dbp_pred_l2", 'rb'))
pred_semantic_l2_pad = pl.load(open(r'../dataset/dbp/predictlabel/dbp_pred_semantic_l2_pad', 'rb'))
########################################################################################################################

x_train, x_test, y1_train, y1_test, y2_train, y2_test, y3_train, y3_test, true_semantic_l2_train, true_semantic_l2_test,\
    pred_l1_train, pred_l1_test, pred_l2_train, pred_l2_test= \
    train_test_split(x, y1, y2, y3, y2_pad[:,:3], pred_l1, pred_l2, test_size=0.2, random_state=42)

emb_label_train = list(np.column_stack((true_semantic_l2_train,x_train)))
emb_label_test = list(np.column_stack((true_semantic_l2_test,x_test)))

#################################
print('Build model...')
model = TextONLSTM3(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../log/dbp/dbp_l2_weights_with_true.hdf5", by_name=True)
#########################
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
model.summary()
print('Train...')
fileweights = r"../log/dbp/dbp_l3_weights_with_true.hdf5"
if os.path.exists(fileweights):
    model.load_weights(fileweights)
checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=5, mode='auto')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', verbose=1, factor=0.1, patience=3, mode='auto')
# model.fit([emb_label_train], y3_train,
#           validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#
#           verbose=2,
#           callbacks=[early_stopping, checkpoint, reduce_lr],
#           shuffle= True)

loss,accuracy = model.evaluate([emb_label_test], y3_test, verbose=0)
print('\ntest loss',loss,'accuracy',accuracy)

print('level3: end up')

pred_l3 = model.predict([emb_label_test], batch_size=batch_size)#此处传入的X_test和Input层的内容一致，要是多个Input()，就传个列表，和model.fit传入的参数一致(不传y_test)


one_hot_pred_l1 = pred_l1_test
one_hot_pred_l2 = trans_to_onehot(pred_l2_test)
one_hot_pred_l3 = trans_to_onehot(pred_l3)
one_hot_truth_l1 = y1_test
one_hot_truth_l2 = y2_test
one_hot_truth_l3 = y3_test

pred_l3 = np.argmax(pred_l3, axis=1)
y3_test = np.argmax(y3_test, axis=1)

print(classification_report(y3_test, pred_l3, digits=8))

#####multilabel metrics in final layer######################

print("coverage_error at the highest levels:\n",coverage_error(one_hot_truth_l3, one_hot_pred_l3))
print("label_ranking_loss at the highest levels:\n",label_ranking_loss(one_hot_truth_l3, one_hot_pred_l3))
print("label_ranking_average_precision_score at the highest levels:\n", label_ranking_average_precision_score(one_hot_truth_l3, one_hot_pred_l3))
print("dcg_score at the highest levels:\n",dcg_score(one_hot_truth_l3, one_hot_pred_l3))
print("ndcg_score at the highest levels:\n",ndcg_score(one_hot_truth_l3, one_hot_pred_l3))

truth = np.column_stack((one_hot_truth_l1, one_hot_truth_l2, one_hot_truth_l3))
pred = np.column_stack((one_hot_pred_l1, one_hot_pred_l2, one_hot_pred_l3))

print("coverage_error:\n", coverage_error(truth, pred))
print("label_ranking_loss:\n", label_ranking_loss(truth, pred))
print("label_ranking_average_precision_score:\n", label_ranking_average_precision_score(truth, pred))
print("dcg_score:\n", dcg_score(truth, pred))
print("ndcg_score:\n", ndcg_score(truth, pred))
print("hamming_loss:\n", hamming_loss(truth, pred))


