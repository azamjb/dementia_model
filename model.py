
# Supervised learning model (labeled data)
# Binary classification model

# Random Forest Classifier (handles tabular, engineered features very well, works well with small to medium-sized datasets)

# sklearn library model to be used

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

X = []
y = [0,1] # dementia, no dementia

# need to store data here ^

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # Splitting data (train, validation split)


# Training the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)


# Predicting on the test set
y_pred = model.predict(X_test)