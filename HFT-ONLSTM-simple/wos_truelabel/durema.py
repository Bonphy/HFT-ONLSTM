import pickle as pl
import numpy as np
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(sparse=False)
pred_l1 = pl.load(open(r"../dataset/wos/predictlabel/wos_pred_l1", 'rb'))
#一个train_data含有多个特征，使用OneHotEncoder时，特征和标签都要按列存放, sklearn都要用二维矩阵的方式存放
one_hot_train_label2 = enc.fit_transform(pred_l1.reshape(-1, 1))   # 如果不加 toarray() 的话，输出的是稀疏的存储格式，即索引加值的形式，也可以通过参数指定 sparse = False 来达到同样的效果
one_hot_train_label1 = one_hot_train_label2[:,:-2]
print(one_hot_train_label2)
# k = [i[0] + i[1] for i in zip(one_hot_train_label1, one_hot_train_label2)]
k = np.append(one_hot_train_label1,one_hot_train_label2,axis=1)
print(k)