# 🏦 Loan Approval Prediction using Machine Learning

A Machine Learning project that predicts loan approval using Logistic Regression and K-Nearest Neighbors (KNN), with data preprocessing, visualization, and accuracy evaluation.

## Overview

This project predicts whether a loan application will be approved or rejected using Machine Learning classification algorithms.

The project uses a loan dataset containing information about applicants and their loan applications. The data is preprocessed by removing the `Loan_ID` column, encoding categorical features, and handling missing values.

Two Machine Learning classification models are trained and evaluated in this project:

* Logistic Regression
* K-Nearest Neighbors (KNN)

## Features

* Loan approval prediction
* Data preprocessing
* Categorical data encoding
* Missing value handling
* Correlation analysis
* Logistic Regression
* K-Nearest Neighbors (KNN)
* Accuracy evaluation

## Dataset

The project uses a loan dataset stored in:

```text
loan.csv
```

The dataset contains information about loan applicants and their applications.

The target variable is:

```text
Loan_Status
```

which represents whether the loan application was approved or rejected.


## How to Run

1. Make sure Python is installed.
2. Place `loan.csv` in the project directory.
3. Install the required libraries.
4. Run the Python script:

```bash
python loan_prediction.py
```

The program will display the correlation heatmap and print the accuracy scores of Logistic Regression and KNN.

## Requirements

The main dependencies are:

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
```

## Contributing

Contributions, suggestions, and bug reports are welcome.

Feel free to fork this repository and submit a pull request.

## License

This project is licensed under the MIT License.

## Author

**Mohammad Reza Bakhshandeh**

Electrical Engineering (Electronics) Graduate

Interested in Python Development, Machine Learning, Deep Learning, Computer Vision, Artificial Intelligence, and Game Development.
