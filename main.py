import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()

import pandas as pd
df = pd.read_csv('parkinsons.csv')
df.head()

selected_features = ['MDVP:Fhi(Hz)', 'D2']
target_feature = 'status'

import numpy as np
from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()
df[selected_features] = min_max_scaler.fit_transform(df[selected_features])

from sklearn.model_selection import train_test_split
X = df[selected_features]
y = df[target_feature]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
model = SVC(kernel='rbf', C=1, gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
accuracy


