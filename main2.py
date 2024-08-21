import streamlit as st
import seaborn as sns
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

x = np.random.rand(100)
y = 2 * x + 1 + 0.2* np.random.rand(100)
plt. scatter(x, y)

st.sidebar.title('Classifiers')
classifier = st.sidebar.selectbox('Select Classifier', ('KNN', 'SVM','RF', 'DT','NN'))
k = st.sidebar.slider('K', 1, 20, 3)

if classifier == 'KNN':
  knn = KNeighborsRegressor(n_neighbors=5)
  knn.fit(x.reshape(-1, 1), y)
  y_pred = knn.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'SVM':
  svm = SVR()
  svm.fit(x.reshape(-1, 1), y)
  y_pred = svm.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'RF':
  rf = RandomForestRegressor()
  rf.fit(x.reshape(-1, 1), y)
  y_pred = rf.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'DT':
  dt = DecisionTreeRegressor()
  dt.fit(x.reshape(-1, 1), y))
  y_pred = dt.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
  
if classifier == 'NN':
  nn = MLPRegressor()
  nn.fit(x.reshape(-1, 1), y))
  y_pred = nn.predict(x.reshape(-1, 1))
  fig, ax = plt.subplots()
  ax.scatter(x, y)
  ax.scatter(x, y_pred)
  st.pyplot(fig)
