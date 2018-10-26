from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from sklearn.model_selection import GridSearchCV
import numpy as np
import csv
import pandas as pd

np.random.seed(7) #fixar a semente pra sempre obter os mesmos resultados
#N sei se funciona perfeitamente com a train_test_split
from tensorflow.python.client import device_lib


reader = csv.reader(open('data.csv','r'), delimiter=',')

rows = np.array(list(reader))
labels = rows[1, 1:-1]
X = rows[2:-1, 1:-1]
Y = rows[2:-1, -1]
#print(rows)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y)
X_train, X_validation, Y_train, Y_validation = train_test_split(X_train, Y_train, test_size=0.2, stratify=Y_train)

########## TESTE ################################

model = Sequential()
qnt_entradas = len(labels)
model.add(Dense(10, input_dim=qnt_entradas, init='normal', activation='relu'))

model.add(Dense(1, init='normal', activation='sigmoid'))



model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #adam = descida do gradiente

checkpoint = ModelCheckpoint("batata.bin", monitor='acc', verbose=1, save_best_only=True, mode='max')

# check 5 epochs
early_stop = EarlyStopping(monitor='acc', patience=5, mode='max')

callbacks_list = [checkpoint, early_stop]
parameters = {'nb_epoch': range(1, 150, 10), 'batch_size':range(10, 200, 10)}
#clf = GridSearchCV(svc, parameters, cv=5)

model.fit(X_train, Y_train, nb_epoch=50, batch_size=10, callbacks=callbacks_list)

scores = model.evaluate(X_validation, Y_validation)

print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
print(qnt_entradas)
#train_test_split()
#print(labels)
