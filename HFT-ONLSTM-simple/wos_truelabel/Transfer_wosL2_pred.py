# coding=utf-8
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle as pl
import numpy as np
from sklearn.metrics import *
# from sklearn.metrics import classification_report, coverage_error, label_ranking_loss, \
#     label_ranking_average_precision_score, dcg_score, ndcg_score, hamming_loss, precision_score
from sklearn.model_selection import train_test_split
from keras.callbacks import ReduceLROnPlateau
from wos_textONlstmL2 import TextONLSTM2

# from sklearn.preprocessing import OneHotEncoder
from dbp_process.probabilityonehot import trans_to_onehot

maxlen = 502
# maxlen = 510
max_features = 89098
batch_size = 64
embedding_dims = 300
epochs = 100
#  The path of embedding matrix
pretrained_w2v, _, _ = pl.load(open(r'../embeddings/emb_matrix_glove_300', 'rb'))
#######################################################################################################################
print('Loading data...')
#  Path to the processed dataset
x,y1,y2,y1_pad,y2_pad = pl.load(open(r'../dataset/wos/WOSDATA_txt_vector500dimsy1y2_10dim_zjp','rb'))


pred_l1 = pl.load(open(r"../dataset/wos/predictlabel/wos_pred_l1", 'rb'))
pred_semantic_l1_pad = pl.load(open(r'../dataset\wos\predictlabel\wos_pred_semantic_l1_pad','rb'))

x_train, x_test, y1_train, y1_test, y2_train, y2_test, pred_semantic_l1_train, pred_semantic_l1_test,\
    pred_l1_train, pred_l1_test= \
    train_test_split(x, y1, y2, pred_semantic_l1_pad, pred_l1, test_size=0.2, random_state=42)

# x_train, x_test, y2_train, y2_test = train_test_split( x, y2, test_size=0.2, random_state=42)
# # l1_count = y1_test.shape[1]
# # l2_count = y2_test.shape[1]
# # label_count = l1_count + l2_count
# # x_train,x_test,pred_semantic_l1_train,pred_semantic_l1_test=train_test_split( x, pred_semantic_l1_pad, test_size=0.2, random_state=42)
# x_train,x_test,pred_semantic_l1_train,pred_semantic_l1_test=train_test_split( x, pred_semantic_l1_pad, test_size=0.2, random_state=42)
# #嵌入真实父标签###true label###########################################################################################
# x_train, x_test, y1_train_pad, y1_test_pad = train_test_split( x, y1_pad, test_size=0.2, random_state=42)

# emb_label_train = list(np.column_stack((y1_train_pad,x_train)))
# emb_label_test = list(np.column_stack((y1_test_pad,x_test)))
###################predicted label#####################################################################################
# 嵌入预测父标签
emb_label_train = list(np.column_stack((pred_semantic_l1_train,x_train)))
emb_label_test = list(np.column_stack((pred_semantic_l1_test,x_test)))
###############################################################################################1024####################
model = TextONLSTM2(maxlen, max_features, embedding_dims, pretrained_w2v).get_model()
model.load_weights(r"../log/wos/wos_weight_l1.hdf5", by_name=True)
model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
print("模型层数：")
print(len(model.layers))
print(model.layers[-4].name)
print("###############################################################################################################")
# 可训练层
for x in model.trainable_weights:
    print(x.name)
print('\n')
# 不可训练层
for x in model.non_trainable_weights:
    print(x.name)
print('\n')
model.summary()
#######################################################################################################################
print('Train...')
fileweights = r"../log/wos/wos_weight_l2.hdf5"
# checkpoint = ModelCheckpoint(fileweights, monitor='val_accuracy', verbose=1, save_best_only=True, mode='auto')
# early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, mode='auto')
# reduce_lr = ReduceLROnPlateau(monitor='val_loss',factor=0.1, patience=3, mode='auto')
# model.fit([emb_label_train], y2_train,
#           validation_split=0.1,
#           batch_size=batch_size,
#           epochs=epochs,
#           callbacks=[early_stopping, checkpoint, reduce_lr],
#           # validation_data=([emb_label_test], y2_test),
#           shuffle= True,
#           verbose= 2)
model.load_weights(fileweights, by_name=True)
loss,accuracy = model.evaluate([emb_label_test], y2_test, verbose=0)
print('\ntest loss',loss,'accuracy',accuracy)
pred_l2=model.predict([emb_label_test], batch_size=batch_size)#此处传入的X_test和Input层的内容一致，要是多个Input()，就传个列表，和model.fit传入的参数一致(不传y_test)


one_hot_pred_l1 = pred_l1_test
one_hot_pred_l2 = trans_to_onehot(pred_l2)
one_hot_truth_l1 = y1_test
one_hot_truth_l2 = y2_test

pred_l2 = np.argmax(pred_l2, axis=1)
y2_test = np.argmax(y2_test, axis=1)

print(classification_report(y2_test, pred_l2))


#####multilabel metrics######################

# enc = OneHotEncoder(sparse=False)
#一个train_data含有多个特征，使用OneHotEncoder时，特征和标签都要按列存放, sklearn都要用二维矩阵的方式存放
   # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果


# pred_l1 = pl.load(open(r"../dataset/wos/predictlabel/wos_pred_l1", 'rb'))
# one_hot_pred_l1 = enc.fit_transform(pred_l1.reshape(-1, 1))
# pred_l2 = pred_l2
# one_hot_pred_l2 = enc.fit_transform(pred_l2.reshape(-1, 1))
pred = np.append(one_hot_pred_l1, one_hot_pred_l2, axis=1)
# truth_l1 = y1_test
# one_hot_truth_l1 = truth_l1
# truth_l2 = y2_test
# one_hot_truth_l2 = enc.fit_transform(truth_l2.reshape(-1, 1))
#
truth = np.append(one_hot_truth_l1, one_hot_truth_l2, axis=1)

print("hamming loss:\n",hamming_loss(truth, pred))
print("accuracy score:\n",accuracy_score(truth, pred))
print("multilabel confusion matrix:\n",multilabel_confusion_matrix(truth, pred))
print("classification_report:\n",classification_report(truth, pred))
print("precision_score:\n",precision_score(truth, pred,average='samples'))
print("recall_score:\n",recall_score(truth, pred,average='samples'))
print("f1_score:\n",f1_score(truth, pred,average='samples'))
print("jaccard_score:\n",jaccard_score(truth, pred,average='samples'))







