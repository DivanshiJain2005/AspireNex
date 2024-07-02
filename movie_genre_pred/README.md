# Movie Genre Prediction

## Overview
This project aims to predict movie genres based on their descriptions using machine learning techniques. The model is built using the TfidfVectorizer for feature extraction and Multinomial Naive Bayes for classification.

## Dataset
The dataset consists of movie descriptions and their corresponding genres. It is split into training and testing sets to evaluate the model's performance.

### Files
- `train_data.txt`: Training dataset containing movie descriptions and genres.
- `test_data.txt`: Testing dataset containing movie descriptions.
- `test_data_solution.txt`: Testing dataset solutions containing genres.

## Project Structure
- `Movie_Genre_Prediction.ipynb`: Jupyter notebook containing the data processing, model training, and evaluation.
- `README.md`: This README file.
- `movie_genre_model.pkl`: Saved model file for future predictions.

## Requirements
To run the code in this project, you will need the following libraries:
- numpy
- pandas
- scikit-learn
- joblib

You can install these libraries using pip:
```bash
pip install numpy pandas scikit-learn joblib
```

## Instructions

### 1. Load the Dataset
Load the training and testing datasets using pandas.

### 2. Feature Extraction
Use TfidfVectorizer to convert the movie descriptions into a matrix of TF-IDF features.

### 3. Train the Model
Train a Multinomial Naive Bayes classifier on the TF-IDF features.

### 4. Evaluate the Model
Evaluate the model's performance on the training set using various metrics.

#### Classification Report on Training Set
```
              precision    recall  f1-score   support

      action       0.70      0.09      0.16      1315
       adult       0.79      0.05      0.10       590
   adventure       0.76      0.05      0.10       775
   animation       0.00      0.00      0.00       498
   biography       0.00      0.00      0.00       265
      comedy       0.56      0.45      0.50      7447
       crime       0.00      0.00      0.00       505
 documentary       0.57      0.90      0.70     13096
       drama       0.47      0.84      0.60     13613
      family       1.00      0.00      0.01       784
     fantasy       0.00      0.00      0.00       323
   game-show       1.00      0.14      0.24       194
     history       0.00      0.00      0.00       243
      horror       0.78      0.36      0.50      2204
       music       0.90      0.16      0.27       731
     musical       0.00      0.00      0.00       277
     mystery       0.00      0.00      0.00       319
        news       0.00      0.00      0.00       181
  reality-tv       0.85      0.03      0.05       884
     romance       0.00      0.00      0.00       672
      sci-fi       0.85      0.04      0.09       647
       short       0.66      0.11      0.19      5073
       sport       0.80      0.11      0.19       432
   talk-show       1.00      0.01      0.02       391
    thriller       0.71      0.02      0.05      1591
         war       0.00      0.00      0.00       132
     western       0.97      0.59      0.73      1032

     accuracy                           0.54     54214
    macro avg       0.50      0.15      0.17     54214
 weighted avg       0.57      0.54      0.46     54214
```

### 5. Save the Model
Save the trained model using joblib.

### 6. Load the Model
Load the saved model for future predictions.

### 7. Make Predictions
Use the loaded model to make predictions on new movie descriptions.

## Conclusion
This project demonstrates the application of TF-IDF and Multinomial Naive Bayes for movie genre prediction based on descriptions. Despite achieving moderate accuracy, the model can be improved by exploring different algorithms, tuning hyperparameters, and enhancing feature engineering.
