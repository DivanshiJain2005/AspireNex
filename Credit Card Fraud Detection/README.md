# Credit Card Fraud Detection

## Overview
This project demonstrates the use of machine learning techniques to detect fraudulent credit card transactions. Using a dataset from Kaggle, we build and evaluate a Random Forest Classifier to identify fraudulent transactions. The dataset contains anonymized transaction features, and our goal is to achieve high accuracy and reliability in fraud detection.

## Dataset
The dataset used in this project is the MLG-ULB Credit Card Fraud Detection dataset available on Kaggle. It contains 284,807 transactions, each described by 31 features including time, amount, and 28 anonymized variables (V1-V28). The target variable 'Class' indicates whether a transaction is valid (0) or fraudulent (1).

- **Dataset URL**: [Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Project Structure
The project is structured as follows:

- **creditcard.csv**: The dataset file.
- **Credit_Card_Fraud_Detection.ipynb**: Jupyter notebook containing the data analysis, model training, and evaluation.
- **credit_fraud_model.pkl**: Trained Random Forest model saved using joblib.
- **README.md**: This README file.

## Requirements
To run the code in this project, you'll need the following libraries:

- numpy
- pandas
- matplotlib
- seaborn
- scikit-learn
- joblib

You can install these libraries using pip:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn joblib
```

## Instructions

### 1. Download the Dataset
Download the dataset from Kaggle and place the `creditcard.csv` file in the project directory.

### 2. Run the Jupyter Notebook
Open the `Credit_Card_Fraud_Detection.ipynb` notebook and run each cell to execute the code. The notebook includes:

1. **Data Loading and Exploration**: Load the dataset and perform initial data exploration.
2. **Data Preprocessing**: Handle missing values and standardize features.
3. **Model Training**: Train a Random Forest Classifier on the dataset.
4. **Model Evaluation**: Evaluate the model using various performance metrics.
5. **Model Saving and Loading**: Save the trained model and demonstrate loading it for predictions.

### 3. Evaluate the Model
The notebook evaluates the model using accuracy, precision, recall, F1-score, and Matthews Correlation Coefficient to ensure robust performance.

### 4. Make Predictions
You can use the trained model to make predictions on new data by loading the saved model and using it to predict the class of new transactions.

## Results
The Random Forest Classifier achieved the following performance metrics on the test set:

- **Accuracy**: 0.9996
- **Precision**: 0.9747
- **Recall**: 0.7857
- **F1-Score**: 0.8701
- **Matthews Correlation Coefficient**: 0.8749

These results demonstrate the model's ability to accurately detect fraudulent transactions, though further improvements can be explored to enhance its sensitivity to fraud cases.

## Conclusion
This project showcases the application of machine learning to detect credit card fraud, highlighting the importance of using robust models to identify fraudulent transactions. Future work could focus on exploring other algorithms, improving feature engineering, and handling class imbalance more effectively.

## Acknowledgements
- The dataset used in this project was provided by Kaggle and the Machine Learning Group at Université Libre de Bruxelles (ULB).
- Special thanks to the contributors of the dataset and the Kaggle community for providing a platform for data science collaboration.
