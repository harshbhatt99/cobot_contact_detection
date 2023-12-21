from joblib import dump, load
from flask import Flask, request, jsonify

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

# Creating Flask object
app = Flask(__name__)

# Load the trained model
model = load('C:\\aiprojects\\machine-learning\\cobot_contact_detection\\cobot_collision_detection.joblib')

def data_preprocessing(df_test):
    X_test = df_test.drop('Label', axis=1)
    y_test = pd.DataFrame(df_test['Label'], columns=['Label'])

    # Encoding the labels (5 categories - label 0 to 4)
    number = LabelEncoder()
    test_labels = number.fit_transform(y_test['Label'].ravel())

    return X_test, test_labels

def feature_selection(data_df):
    def extract_features(data, start_col, end_col):
        features = []
        for i in range(28):
            subset = data.iloc[:, start_col + 7 * i:end_col + 1 + 7 * i]
            features.append(subset.mean(axis=1))
            features.append(subset.std(axis=1))
            features.append(subset.max(axis=1))
            features.append(subset.min(axis=1))
        return pd.concat(features, axis=1)
    
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

@app.route('/predict', methods=['POST'])

def prediction():
    json_data = request.get_json()  # Get data sent to the endpoint
    if isinstance(json_data, dict):
        # Single observation
        sample_data = pd.DataFrame([json_data])
    elif isinstance(json_data, list):
        # Multiple observations
        sample_data = pd.DataFrame(json_data)

    sample_features = feature_selection(sample_data)
    sample_scaled = data_normaliaztion(sample_features)

    prediction = model.predict(sample_scaled)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
