import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
import numpy as np 
import pandas as pd
import joblib

df = pd.read_csv("/Users/rayyanwaseem/Desktop/Projects/Loan-Classification-Neural-Network/loan_data.csv")
#print(df.sample(5))

#print(df.isna().sum())


#Numerical Features 
num_features = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

#Categorical Features 
cat_features = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

feature_cols = num_features + cat_features
target_col = 'loan_status'


#You can see that people who are 144 years old are eligible to take out a loan
#print(df['person_age'].max())

#We can assume that people over the age of 70 will not be taking out any loans.
#In this case we can exclude them
df[df['person_age'] > 70]['person_age'].count()
df = df[df['person_age']<70]

#print(df['person_age'].max())

#Now we will define our two variables X and y 
X = df[feature_cols]
y = df[target_col]

#One-hot encode the categorical features
X = pd.get_dummies(X, columns=cat_features, drop_first=True)
model_input_columns = X.columns.tolist()  # Save the model input columns for later use
joblib.dump(model_input_columns, 'model_input_columns.pkl')  # Save the model input columns
print("Model input columns saved as model_input_columns.pkl") # Save the model input columns for future use
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size= 0.2, stratify = y)

scaler = StandardScaler().fit(X_train)
joblib.dump(scaler, 'scaler.pkl') 
print("Scaler saved as scaler.pkl") # Save the scaler for future use

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype = torch.float32)
X_test_tensor = torch.tensor(X_test, dtype = torch.float32)

y_train_tensor = torch.tensor(y_train.values, dtype = torch.float32).unsqueeze(1) #Add a dimension to make it [batch_size, 1]
y_test_tensor = torch.tensor(y_test.values, dtype = torch.float32).unsqueeze(1) #Add a dimension to make it [batch_size, 1]

#Now that the data has been preprocessed and initialized as a tensor, we can build our NN model
input_dim = X_train.shape[1]

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

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0027)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 10)

num_epoch = 200
history = {"Loss": [], "Validation Loss": []}
for epoch in range(num_epoch):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor) #Now the shapes should match 
    loss.backward()
    optimizer.step()
    history['Loss'].append(loss.item())

    model.eval()
    with torch.no_grad():
        outputs_val = model(X_test_tensor)
        test_loss = criterion(outputs_val, y_test_tensor).item() #Calculate cross entropy loss
        val_loss = criterion(outputs_val, y_test_tensor) #Calculate validation loss #Shapes should match here too
        predicted = (torch.sigmoid(outputs_val) > 0.5).float() #ensures proper binary classification  
        test_accuracy = accuracy_score(y_test_tensor, predicted)
        history['Validation Loss'].append(val_loss.item())

    scheduler.step(val_loss)

    if (epoch+1) % 10 == 0:
        lr = optimizer.param_groups[0]['lr']
        print(f"Epoch [{epoch+1}/{num_epoch}], Loss: {loss.item():.4f}, Validation Loss: {val_loss.item():.4f}, LR: {lr:.6f}")
        print(f"Test Accuracy: {test_accuracy:.4f}")

# Save the model
torch.save(model.state_dict(), 'loan_classification_model.pth')
print("Model saved as 'loan_classification_model.pth'")