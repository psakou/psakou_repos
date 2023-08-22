#Import the necessary librairies
import pandas as pd
import numpy as np
import streamlit as st
import sklearn

#As we're going to load the datasets
from sklearn import datasets
from sklearn import ensemble

#Let's load the IRIS Dataset
i_data = datasets.load_iris()

print(i_data.target_names)

data=pd.DataFrame({
'sepal length': i_data.data[:,0],
'sepal width': i_data.data[:,1],
'petal length': i_data.data[:,2],
'petal width': i_data.data[:,3],
'Species': i_data.target})

X = i_data.data #features
Y = i_data.target #target

from sklearn.ensemble import RandomForestClassifier #Importing Random Forest Classifier
from sklearn import metrics  # Importing metrics to test accuracy
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test= train_test_split(X, Y, test_size=0.3) #splitting data with test size of 30%

#RF Prediction
clf=RandomForestClassifier(n_estimators=3)  #Creating a random forest with 3 decision trees
clf.fit(x_train, y_train)  #Training our model
y_pred=clf.predict(x_test)  #testing our model
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))  #Measuring the accuracy of our model

#Title
st.title("Iris")

#Header
st.header("Iris species")

#Let's know min, max and mean for each feature
data.describe()

# Using the "streamlit.slider()" function
# Using the minimum, maximum, and mean values of each feature as the arguments for the function.

# Add input fields for sepal length
slinfo = st.slider("Info longueur sépales", 4.300000, 7.900000, 5.843333)

# Add input fields for sepal width
swinfo = st.slider("Info largeur sépales", 2.000000, 4.400000, 3.057333)

# Add input fields for petal length
plinfo = st.slider("Info longueur pétales", 1.000000, 6.900000, 3.758000)

# Add input fields for petal width
pwinfo = st.slider("Info largeur pétales", 0.100000, 2.500000, 1.199333)

# Bouton de prédiction
if st.button("Prédire"):
    # Créer un tableau avec les valeurs d'entrée
    input_data = [[slinfo, swinfo, plinfo, pwinfo]]
    
    # Effectuer la prédiction
    prediction = clf.predict(input_data)
    predicted_class = i_data.target_names[prediction[0]]
    
    # Afficher le résultat de la prédiction
    st.write("Le type de fleur d'Iris prédit est :", predicted_class)  