from keras.models import Sequential
from sklearn.model_selection import train_test_split
import numpy as np
import csv

reader = csv.reader(open('subdata.csv','r'), delimiter=',')

rows = np.array(list(reader))
labels = rows[0]
print(labels)
X = rows[2:-1, 1:-1]
Y = rows[2:-1, -1]
#print(rows)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)



#print(X_train)
#train_test_split()
#print(labels)
