# Small debugging to allow sklearn importing
import sys
sys.path.append('/usr/local/lib/python3.8/site-packages')
# -------------------------
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from scipy.sparse import hstack

df = pd.read_csv('data/Shakespeare_data.csv')

 ## Cleaning Data 

df.head(10)
df.drop(columns = ["Dataline"], inplace = True)
df = df.query('not Player.isnull()')
df = df.query('not ActSceneLine.isnull()')

temp = df["ActSceneLine"].str.split(".", n = 2, expand = True)
df["Act"]= temp[0] 
df["Scene"]= temp[1] 
df["Line"]= temp[2] 
df.drop(columns =["ActSceneLine"], inplace = True)
df.head(10)

le = preprocessing.LabelEncoder()
df['player_le'] = le.fit_transform(df['Player'])
df['play_le'] = le.fit_transform(df['Play'])
df['player_line_le'] = le.fit_transform(df['PlayerLine'])
df['act_le'] = le.fit_transform(df["Act"])
df['scene_le'] = le.fit_transform(df["Scene"])
df['line_le'] = le.fit_transform(df["Line"])
df

# Feature Engineering

### Decision Tree
x = df[['play_le', 'act_le', 'scene_le', 'line_le']]
y = df['player_le']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=1)

tree = DecisionTreeClassifier()
tree_classifier = tree.fit(x_train, y_train)
tree_pred = tree_classifier.predict(x_test)
print("Prediction Accuracy:", metrics.accuracy_score(y_test, tree_pred))

### Random Forest
forest = RandomForestClassifier()
forest.fit(x_train, y_train)
forest_pred = forest.predict(x_test)
print("Prediction Accuracy:", metrics.accuracy_score(y_test, forest_pred))