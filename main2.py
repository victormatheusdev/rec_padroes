from keras.models import Sequential
from sklearn.model_selection import train_test_split
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
import numpy as np
import csv
from tensorflow.python.client import device_lib
reader = csv.reader(open('data.csv','r'), delimiter=',')
rows = np.array(list(reader))

labels = rows[1, 1:-1]
qnt_entradas = len(labels)

def create_baseline():
    # create model
    model = Sequential()
    model.add(Dense(10, input_dim=qnt_entradas, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
    # Compile model. We use the the logarithmic loss function, and the Adam gradient optimizer.
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model



#N sei se funciona perfeitamente com a train_test_split


X = rows[2:-1, 1:-1]
Y = rows[2:-1, -1]
#print(rows)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, stratify=Y)
########## TESTE ################################

estimators = []
estimators.append(('standardize', StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=create_baseline, epochs=50, batch_size=75, verbose=1)))
pipeline = Pipeline(estimators)
kfold = StratifiedKFold(n_splits=10, shuffle=True)
results = cross_val_score(pipeline, X_train, Y_train, cv=kfold)
print("Results: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
#train_test_split()
#print(labels)
