# Import necessary libraries
import numpy as np
import joblib  # For loading the serialized model
import pandas as pd  # For data manipulation
from flask import Flask, request, jsonify  # For creating the Flask API

# Initialize the Flask application
superkart_api = Flask("SuperKart Sales Predictor")

# Load the trained machine learning model (preprocessing pipeline + model)
model = joblib.load("superkart_model.joblib")

# Features expected by the model (the same features used during training)
FEATURES = [
    'Product_Weight',
    'Product_Sugar_Content',
    'Product_Allocated_Area',
    'Product_MRP',
    'Store_Size',
    'Store_Location_City_Type',
    'Store_Type',
    'Product_Id_char',
    'Store_Age_Years',
    'Product_Type_Category'
]

# Define a route for the home page (GET request)
@superkart_api.get('/')
def home():
    """
    This function handles GET requests to the root URL ('/') of the API.
    It returns a simple welcome message.
    """
    return "Welcome to the SuperKart Sales Prediction API!"

# Define an endpoint for single product-store prediction (POST request)
@superkart_api.post('/v1/predict')
def predict_sales():
    """
    This function handles POST requests to the '/v1/predict' endpoint.
    It expects a JSON payload containing product and store details and returns
    the predicted sales revenue as a JSON response.
    """
    # Get the JSON data from the request body
    sales_data = request.get_json()

    # Return an error if any feature is missing
    missing = [feature for feature in FEATURES if feature not in sales_data]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Extract relevant features from the JSON data
    sample = {feature: sales_data[feature] for feature in FEATURES}

    # Convert the extracted data into a Pandas DataFrame
    input_data = pd.DataFrame([sample])

    # Make prediction
    predicted_sales = model.predict(input_data)[0]

    # Convert predicted_sales to Python float and round it
    predicted_sales = round(float(predicted_sales), 2)
    # The conversion above is needed as the model returns NumPy float values,
    # which Flask's jsonify function cannot serialize directly

    # Return the predicted sales
    return jsonify({'Predicted_Sales': predicted_sales})


# Define an endpoint for batch prediction (POST request)
@superkart_api.post('/v1/predictbatch')
def predict_sales_batch():
    """
    This function handles POST requests to the '/v1/predictbatch' endpoint.
    It expects a CSV file containing product and store details for multiple rows
    and returns the predicted sales as a dictionary in the JSON response.
    """
    # Get the uploaded CSV file from the request
    file = request.files['file']

    # Read the CSV file into a Pandas DataFrame
    input_data = pd.read_csv(file)

    # Return an error if any feature is missing
    missing = [feature for feature in FEATURES if feature not in input_data.columns]
    if missing:
        return jsonify({'error': f'Missing features: {missing}'}), 400

    # Make predictions for all rows in the DataFrame
    predicted_sales = [round(float(sales), 2) for sales in model.predict(input_data[FEATURES])]

    # Create a dictionary of predictions with the row index as keys
    output_dict = {str(index): sales for index, sales in zip(input_data.index, predicted_sales)}

    # Return the predictions dictionary as a JSON response
    return jsonify(output_dict)

# Run the Flask application in debug mode if this script is executed directly
if __name__ == '__main__':
    superkart_api.run(host='0.0.0.0', port=7860, debug=True)
