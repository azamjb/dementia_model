# Supervised learning model (labeled data)
# Binary classification model

# Random Forest Classifier (handles tabular, engineered features very well, works well with small to medium-sized datasets)

# sklearn library to be used

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
import sys

# Redirect all output to a file

with open("model1_results.txt", "w") as f:

    sys.stdout = f  # Redirect stdout to the file

    df = pd.read_csv("model3_syntactic.csv") 

    # Loading CSV data into a DataFrame ^


    feature_cols = ["parse_tree_depth","subordinate_clause_ratio","pos_bigram_entropy","avg_dependency_distance"] # - syntactic 

    # feature_cols = ["pronoun_to_noun_ratio", "filler_word_count","correction_phrase_count","conjunction_overuse"] # - pragmatic 

    # feature_cols = ["sentence_length", "word_count", "type_token_ratio", "avg_word_length"] - baseline

    # Selecting columns from dataset to use as featues (uncomment the one to be used) ^


    X = df[feature_cols]

    # Extracing the features from datast into our input matrix ^


    le = LabelEncoder()
    y = le.fit_transform(df["label"]) 

    # Encoding the label column into numbers (1 for alzheimers, 0 for control) ^


    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Split the dataset, 80% for training, 20% for validation, stratifying to keep class distribution equal in both sets ^


    rf = RandomForestClassifier(n_estimators=100, random_state=42)

    # Creating a Random Forest with 100 trees, random state of 42 to ensure reproducibility


    cv_scores = cross_val_score(rf, X_train, y_train, cv=5)
    print(f"5-Fold CV accuracy (train): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Performing 5-fold cross-validation on the training set ^


    rf.fit(X_train, y_train)

    # Training on the full training set ^ 


    y_pred = rf.predict(X_test)

    # Using the trained model to predict on the validation set ^


    print("\nClassification report on test set:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Outputting the precision, recall, F1-score, and support for each class ^


    importances = rf.feature_importances_
    for feat, imp in zip(feature_cols, importances):
        print(f"Feature: {feat:15} Importance: {imp:.4f}")

    # Outputting the feature importance ^


sys.stdout = sys.__stdout__
