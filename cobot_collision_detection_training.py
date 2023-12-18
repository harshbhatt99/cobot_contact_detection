import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import SelectKBest, chi2, SelectFromModel

from sklearn.datasets import make_blobs
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.ensemble import VotingClassifier, StackingClassifier, BaggingClassifier, AdaBoostClassifier

from joblib import dump, load

# From different experiments with algorithms and ensemble methods, it is derived ...
# ... that Random Forest is the most suitable algorithm for training

def load_data(train_path, test_path):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def rename_columns(original_df):
    current_column_names = []
    # Creating column names in the format given in csv file
    for i in range(1,786):
        current_name = [f"Var{i}"]
        current_column_names.extend(current_name)
    
    # Creating a list of new column names
    new_column_names = []
    new_column_names.append("Label")
    sample_step = 0
    for i in range (1,29):
        for j in range(1,8):
            new_name = [f"S{i}_Joint_Torque{j}"]
            new_column_names.extend(new_name)
        for k in range(1,8):
            new_name = [f"S{i}_External_Torque{k}"]
            new_column_names.extend(new_name)
        for l in range(1,8):
            new_name = [f"S{i}_Delta_Joint_Position{l}"]
            new_column_names.extend(new_name)
        for m in range(1,8):
            new_name = [f"S{i}_Delta_Joint_Velocity{m}"]
            new_column_names.extend(new_name)
    
    # Creating dictionary with new column names for 
    new_column_dict = {}
    for i in range(0,785):
        index = current_column_names[i]
        value = new_column_names[i]
        new_column_dict[index] = value
    
    # Finally, renaming the dataframes
    original_df.rename(columns=new_column_dict, inplace=True)
    
    return original_df

def data_preprocessing(df_train, df_test):
    X_train = df_train.drop('Label', axis=1)
    X_test = df_test.drop('Label', axis=1)
    y_train = pd.DataFrame(df_train['Label'], columns=['Label'])
    y_test = pd.DataFrame(df_test['Label'], columns=['Label'])

    # Encoding the labels (6 categories - label 0 to 5)
    number = LabelEncoder()
    train_labels = number.fit_transform(y_train['Label'].ravel())
    test_labels = number.fit_transform(y_test['Label'].ravel())

    return X_train, X_test, train_labels, test_labels

def feature_selection(data_df):
    def extract_features(data, start_col, end_col):
        features = []
        # When this loop will run for 1 to 7 columns, it will take the mean between ...
        # ... 1, 29, 57, ...
        for i in range(28):
            subset = data.iloc[:, start_col + 7 * i:end_col + 1 + 7 * i]
            features.append(subset.mean(axis=1))
            features.append(subset.std(axis=1))
            features.append(subset.max(axis=1))
            features.append(subset.min(axis=1))
        return pd.concat(features, axis=1)
    
    # Extracting features for training
    joint_torques_features = extract_features(data_df, 1, 7)
    external_torques_features = extract_features(data_df, 8, 14)
    position_diff_features = extract_features(data_df, 15, 21)
    velocity_diff_features = extract_features(data_df, 22, 28)


    features_df = pd.concat([joint_torques_features, external_torques_features, 
                      position_diff_features, velocity_diff_features], axis=1)

    
    return features_df

def data_normaliaztion(features_df):
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_df)
    return features_scaled

def model_training(X_train_scaled, train_labels, X_test_scaled, test_labels):
    # Training the Random Forest classifier with the best parameters
    rf_best = RandomForestClassifier(max_depth=10, min_samples_leaf=1, 
                                    min_samples_split=5, n_estimators=200, random_state=42)
    rf_best.fit(X_train_scaled, train_labels)

    # Predicting on the test set
    y_pred_rf = rf_best.predict(X_test_scaled)

    # Evaluating the model
    accuracy_rf = accuracy_score(test_labels, y_pred_rf)
    recall_rf = recall_score(test_labels, y_pred_rf, average='weighted')
    precision_rf = precision_score(test_labels, y_pred_rf, average='weighted')
    f1_score_rf = f1_score(test_labels, y_pred_rf, average='weighted')

    # print("Accuracy: " , accuracy_rf)
    # print("Recall: " , recall_rf)
    # print("Precision: " , precision_rf)
    # print("F1 Score: " , f1_score_rf)

    dump(rf_best, 'C:\\aiprojects\\machine-learning\\cobot_contact_detection\\cobot_collision_detection.joblib')

    return accuracy_rf, recall_rf, precision_rf, f1_score_rf

def prediction(sample_data):
    sample_data = rename_columns(sample_data)
    sample_features = feature_selection(sample_data)
    sample_scaled = data_normaliaztion(sample_features)

    model = load('C:\\aiprojects\\machine-learning\\cobot_contact_detection\\cobot_collision_detection.joblib')
    sample_prediction = model.predict(sample_scaled)
    return sample_prediction

# Load the data
train_path = "C:\\aiprojects\\machine-learning\\cobot_contact_detection\\contact_detection_train.csv"
test_path = "C:\\aiprojects\\machine-learning\\cobot_contact_detection\\contact_detection_test.csv"
df_train, df_test = load_data(train_path, test_path)

# Rename the columns (optional)
df_train = rename_columns(df_train)
df_test = rename_columns(df_test)

# Preprocess the data
# Split into X and y
# Label the output classes
X_train, X_test, train_labels, test_labels = data_preprocessing(df_train, df_test)

# Feature extraction
# Statistical Features Over Time: Such as rolling averages, standard deviations, or 
# ...more complex statistical measures that summarize the data over time windows.
features_train  = feature_selection(X_train)
features_test =  feature_selection(X_test)

# Scaling the data
X_train_scaled = data_normaliaztion(features_train)
X_test_scaled = data_normaliaztion(features_test)

# Model Training
accuracy_rf, recall_rf, precision_rf, f1_score_rf = model_training(X_train_scaled, train_labels, X_test_scaled, test_labels)

# Prediction
sample_data  = X_test.sample(n=1)
sample_prediction = prediction(sample_data)

if sample_prediction[0] == 0:
    print("No contact")
elif sample_prediction[0] == 1:
    print("Operator grasped 5th link")
elif sample_prediction[0] == 2:
    print("Operator grasped 6th link")
elif sample_prediction[0] == 3:
    print("Alert: 5th link colliding with human")
elif sample_prediction[0] == 4:
    print("Alert: 6th link colliding with human")
else:
    print("Error in predicting")

