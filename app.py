from flask import Flask, request, jsonify
import pickle
import pandas as pd

app = Flask(__name__)

# Load the trained model, scaler, and column names
with open('rf_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open('columns.pkl', 'rb') as columns_file:
    columns = pickle.load(columns_file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame([data])
    
    # Encode categorical variables using one-hot encoding
    df = pd.get_dummies(df, columns=['Vehicle_Type', 'Lane_Type', 'Geographical_Location', 'FastagID'])
    
    # Ensure all expected columns are present
    for col in columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match training data
    df = df[columns]
    
    prediction = model.predict(df)
    return jsonify({'fraud': bool(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
