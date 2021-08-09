import time
import pickle as pl
from nltk.corpus import stopwords
import keras
MAX_LEN = 500
start_time = time.time()
my_not_words = stopwords.words('english')
pretrained_w2v, word_to_id, _ = pl.load(
    open(r'../embeddings/emb_matrix_glove_300', 'rb'))
#################################################################
###########
cont_file = r"./WOS46985/X_clear.txt"
y1_file = r"./WOS46985/YL1_clear.txt"
y2_file = r"./WOS46985/Y_clear.txt"
with open(cont_file, 'r') as f:
    xf = f.readlines()
    x = []
    for xrow in xf:
        x.append([word_to_id[r.lower()] for r in xrow.split() if r.lower() in word_to_id and r.lower() not in my_not_words])

with open(y1_file, 'r') as f:
    yf1 = f.readlines()
    yf1 = [int(i.strip()) for i in yf1]
with open(y2_file, 'r') as f:
    yf2 = f.readlines()
    yf2 = [int(i.strip()) for i in yf2]
l1_sem_set = ['Computer Science', 'Electrical Engineering', 'Psychology', 'Mechanical Engineering', 'Civil Engineering',
      'Medical Sciences', 'Biochemistry']

cont_pad = keras.preprocessing.sequence.pad_sequences(x,MAX_LEN,padding='post', truncating='post')
l1_1hot = keras.utils.to_categorical(yf1)
l2_1hot = keras.utils.to_categorical(yf2)
datafile ='pretrained_wos.pkl'
seml1 = [l1_sem_set[i] for i in yf1]
sem_l1 = []
for temp in seml1:
    sem_l1.append([word_to_id[r.lower()] for r in temp.split() if r.lower() in word_to_id and r.lower() not in my_not_words])
with open(datafile, 'wb') as f:
    pl.dump((cont_pad, l1_1hot, l2_1hot, sem_l1), f)


print("Time cost: %.3f seconds...\n" % (time.time() - start_time))