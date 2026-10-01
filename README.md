# Linear Regression & Gradient Descent from Scratch

## Overview

This project implements **multivariate linear regression from scratch using NumPy**, without using machine-learning libraries such as `scikit-learn`.

The model is trained on the **California Housing Dataset** to predict median house values. The project covers data preprocessing, feature normalization, parameter initialization, prediction, Mean Squared Error (MSE), gradient descent, and visualization of the model's predictions.

This project was completed as part of **Week 2 of the Robotics Club Induction**.

## Features

* Loads and preprocesses the California Housing Dataset
* Removes the categorical `ocean_proximity` feature
* Handles missing values in `total_bedrooms`
* Separates input features and target values
* Normalizes input features using mean and standard deviation
* Implements linear regression from scratch
* Implements Mean Squared Error (MSE)
* Implements gradient descent manually
* Trains the model for a specified number of iterations
* Visualizes actual vs. predicted house values

## Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Google Colab

## Dataset

The project uses the **California Housing Dataset**.

The target variable is:

* `median_house_value`

The following features are used:

* `longitude`
* `latitude`
* `housing_median_age`
* `total_rooms`
* `total_bedrooms`
* `population`
* `households`
* `median_income`

The categorical `ocean_proximity` column is removed during preprocessing.

## Methodology

### 1. Data Preprocessing

The dataset is loaded using Pandas.

The `ocean_proximity` column is removed because the model is implemented using numerical features only.

Missing values in `total_bedrooms` are replaced with the mean value of that column.

### 2. Feature Normalization

Each input feature is standardized using:

```text
x_normalized = (x - mean) / standard_deviation
```

This helps the gradient descent algorithm work more effectively when the features have very different scales.

### 3. Model

The model uses the linear regression equation:

```text
ŷ = Xw + b
```

where:

* `X` = input features
* `w` = model weights
* `b` = bias
* `ŷ` = predicted values

The weights are initially set to zero.

### 4. Cost Function

The model uses **Mean Squared Error (MSE)**:

```text
MSE = (1 / m) Σ(y - ŷ)²
```

where `m` is the number of training examples.

### 5. Gradient Descent

The parameters are updated iteratively using the gradients of the cost function.

The implementation uses:

```text
dw = (-2/m) Xᵀ(y - ŷ)

db = (-2/m) Σ(y - ŷ)
```

The parameters are then updated using the learning rate:

```text
w = w - learning_rate × dw

b = b - learning_rate × db
```

## Training Configuration

The notebook uses:

```text
Learning Rate: 0.01
Iterations: 40,000
Number of Features: 8
```

The cost is printed every 1,000 iterations to monitor the training process.

## Visualization

After training, the notebook generates a scatter plot comparing:

* **Predicted Prices**
* **Actual Prices**

A reference line is also plotted to show where predictions would exactly match the actual values.

## How to Run

### Using Google Colab

1. Open the provided `.ipynb` notebook in Google Colab.
2. Mount Google Drive when prompted.
3. Make sure `housing.csv` is available at the expected location.
4. Run the notebook cells sequentially.

### Using Jupyter Notebook

Install the required dependencies:

```bash
pip install numpy pandas matplotlib
```

Then open the notebook:

```bash
jupyter notebook
```

and run the cells sequentially.

## Project Structure

```text
.
├── Aarush_Jain_week2_(2).ipynb
├── housing.csv
└── README.md
```

## Learning Outcomes

This project demonstrates an understanding of:

* Data preprocessing
* Handling missing data
* NumPy arrays and matrix operations
* Feature normalization
* Linear regression
* Mean Squared Error
* Partial derivatives and gradients
* Gradient descent
* Model evaluation
* Data visualization

## Notes

The model is intentionally implemented from scratch using NumPy rather than using a pre-built machine-learning model. The goal is to understand the mathematical and algorithmic foundations behind linear regression and gradient descent.
