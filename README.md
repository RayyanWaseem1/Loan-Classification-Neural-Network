# Loan-Classification-Neural-Network

## Project Overview
In this project, I developed a feed-forward neural network to classify whether a borrower’s loan will be paid off or in default. Levering a dataset of historical loan applications and outcomes, I created a data pipeline that cleans demographic and financial features, one-hot encodes categorical variables, and standardizes numerical inputs. I then trained a multi-layer perceptron (MLP) with dropout and a learning rate scheduler, optimizing binary cross-entropy loss via the Adam optimizer. The goal of this project was to maximize predictive accuracy on an unseen test set and also deliver a reusable model for deployment in real world applications. 

## Dataset
The raw data comprised loan records loaded in from loan_data.csv, containing the demographics of borrowers (age, gender, education, home ownership), financial metrics (income, employment experience, credit score, credit-bureau history length), and loan-specific attributes (amount requested, interest rate, ratio of loan payment to income, and stated intent). Data points with implausible ages (>70) were removed under the assumption that retirees rarely initiate new loans. This helped in reducing outliers and thus improving model generalizability. After this initial filtering, the final sample included approximately 20,000 loan applications spanning a variety of different risk profiles, with a balanced binary target (loan_status) denoting either “paid” or “default.”

## Machine Learning Methodology
I began by partitioning the cleaned dataset into features and targets, reserving 20% of observations for testing. Categorical predictors were one-hot encoded, with the first level dropped to avoid any collinearity. Numerical predictors were standardized using scikit-learn’s Standard Scaler, both of which were serialized to the disk for reproducibility.

The neural architecture consisted of an input layer matching the preprocessed feature dimensionality. This was followed by three hidden layers of sizes 32, 16, and 8, each with ReLU activations and 30% dropout to mitigate overfitting. The final output node produced a single logit. Training was optimized with BCEWithLogitsLoss objective via the Adam optimizer (learning rate = 0.0027). A ReduceLROnPlateau scheduler was implemented to monitor validation loss in order to adaptively decrease the learning rate upon any stagnation. The model was trained for 200 epochs with full-batch gradient updates, and at every tenth epoch, training loss, validation loss, and test accuracy was logged to keep track of convergence. 

## Results 
Over the course of the training, both the training and validation loss exhibited steady declines, with validation loss plateauing at around 120 epochs - indicating convergence without overfitting. The learning-rate scheduler reduced the step size two times, which facilitate the weight adjustment in later epochs. On the hold-out test set, the neural network achieved a final accuracy of 83.2%, demonstrating robust predictive performance. 

## Lessons Learned 
This project underscored several considerations in credit-risk modelling. First, feature curation - such as filtering by age and encoding based on the intent of the borrower - significantly influenced downstream accuracy and interpretability Second, implementing dropout and the learning-rate scheduler proved essential for balancing the bias-variance tradeoff, particularly with this modestly sized dataset. 

While this static classifier performed well in cross-sectional settings, a production system would benefit from temporal validation (in order to guard against data-leakage), alternative risk metrics (e.g. precision-recall for imbalance defaults), and uncertainty quantification (e.g., via Bayesian neural networks). Also, incorporating transaction level costs and back-testing on rolling windows would further improve this project and align the model with real-world scenarios. 

## Conclusion
Overall, this project illustrated a complete end-to-end pipeline from raw data all the way through model serialization – further demonstrating how deep learning techniques can be tailored to practical credit-risk assessment tasks under real world constraints. Continuous refinement through more granular feature engineering and deployment-oriented monitoring would further strengthen the model’s reliability and overall business value. 
