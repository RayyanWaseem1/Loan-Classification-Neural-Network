from flask import Flask, request, jsonify
from flask_cors import CORS # Import CORS for handling cross-origin requests
import torch 
import torch.nn as nn 
import pandas as pd
import numpy as np 
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model input columns from a file or define them
scaler = joblib.load('scaler.pkl')
model_input_columns = joblib.load('model_input_columns.pkl')

#Defining model architecture
class LoanModel(nn.Module):
    def __init__(self, input_dim):
        super(LoanModel, self).__init__()
        model = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8,1)
        )
                
    def forward(self, x):
        return self.seq(x)
    
#Initialize model with correct input dimension
input_dim = len(model_input_columns)

model = nn.Sequential(
            nn.Linear(input_dim,32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32,16),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(8,1)
        )

model.load_state_dict(torch.load("loan_classification_model.pth", map_location=torch.device('cpu')))
model.eval()  # Set the model to evaluation mode

#Define prediction route
@app.route("/predict", methods = ["POST"])
def predict():
    try:
        data = request.get_json()
        df = pd.DataFrame([data])

        #one hot encode the categorical features (matching training)
        df = pd.get_dummies(df)
        df = df.reindex(columns = model_input_columns, fill_value=0)  # Ensure all columns are present

        #scaling the data
        X_scaled = scaler.transform(df)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        #make the prediction
        with torch.no_grad():
            output = model(X_tensor)
            probability = torch.sigmoid(output).item()  # Get the probability as a scalar
            prediction = int(probability > 0.5)

            return jsonify({
                "probability": probability,
                "prediction": prediction
            })
        
    except Exception as e:
        return jsonify({"error": str(e)})
    
#Run the app
if __name__ == "__main__":
    app.run(debug=True)