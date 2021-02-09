import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import keras_preprocessing

#read the csv dataset file
dataset = pd.read_csv('Churn_Modelling.csv')
print((dataset.head()))

X = dataset.iloc[:, 3:12].values
y = dataset.iloc[:,13].values
print("########################### Orignal X data ##########################\n",X)
print("########################### Orignal y data ##########################\n", y)

print("############################################## Data Preprocessing #####################################################")
# encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder()
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
print("################### After labelencoding_X1 #########################\n", X)
labelencoder_X2 = LabelEncoder()
X[:, 2] = labelencoder_X2.fit_transform(X[:, 2])
print("################### After labelencoding_X2 ##########################\n", X)


from sklearn.compose import ColumnTransformer

columntransformer = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(columntransformer.fit_transform(X))
print("################### After onehot encoding ############################\n", X)
X = X[:, 1:]
print("################## After applying dummy variable trap ################\n", X)

# spliting the data into traning and testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# StandardScaler
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)
print("################### After StandardScaler (X_train) ################## ", X_train)
print("################### After StandardScaler (X_test   #################", X_test)


# Part 2 - Building the ANN

# initializing the ANN
ann = tf.keras.models.Sequential()

# Adding the input layers and first hidden layers
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
# Adding the second layers
ann.add(tf.keras.layers.Dense(units = 6, activation='relu'))
# adding the output layers
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# traning the model
ann.fit(X_train, y_train, batch_size = 32, epochs=100)

# confusion matrics
y_pred = ann.predict(X_test)
print("######################## Y_Prediction (X_test) #########################\n", y_pred)
y_pred = (y_pred > 0.5)
print("######################## Y_Prediction (X_test) #########################\n", y_pred)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("######################### Confusion Matrix ########################\n", cm)
print(" ############################  @shanidahar ######################")
