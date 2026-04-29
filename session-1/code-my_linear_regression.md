# Linear Regression from Scratch

## The Model

Under the **row-vector convention**, the linear regression model for a single sample is:

$$
\hat{y}^{(i)} = x^{(i)} W + b
$$

where:

- $x^{(i)} \in \mathbb{R}^{1 \times d}$ is the input row vector
- $W \in \mathbb{R}^{d \times 1}$ is the weight matrix
- $b \in \mathbb{R}^{1 \times 1}$ is the bias
- $\hat{y}^{(i)}$ is the predicted scalar value

The goal is to learn $W$ and $b$ so that predictions match the true targets $y^{(i)}$ as closely as possible.

## Loss Function

We use **Mean Squared Error (MSE)** to measure prediction error over all $n$ samples:

$$
\mathcal{L}(W, b) = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})^2
$$

For one sample, define the per-example squared error:

$$
\ell^{(i)} = (\hat{y}^{(i)} - y^{(i)})^2
$$

so that $\mathcal{L} = \frac{1}{n} \sum_{i=1}^{n} \ell^{(i)}$.

## Gradient Derivation

Because the loss depends on $W$ through:

$$
W \rightarrow \hat{y}^{(i)} \rightarrow \ell^{(i)}
$$

we apply the chain rule.

### Per-example gradients

**First**, the derivative of the per-example loss w.r.t. the prediction:

$$
\frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} = 2(\hat{y}^{(i)} - y^{(i)})
$$

**Second**, the derivative of the prediction w.r.t. the parameters:

$$
\frac{\partial \hat{y}^{(i)}}{\partial W} = x^{(i)\mathsf{T}}
$$

$$
\frac{\partial \hat{y}^{(i)}}{\partial b} = 1
$$

**Substituting:**

$$
\frac{\partial \ell^{(i)}}{\partial W} = 2 x^{(i)\mathsf{T}} (\hat{y}^{(i)} - y^{(i)})
$$

$$
\frac{\partial \ell^{(i)}}{\partial b} = 2 (\hat{y}^{(i)} - y^{(i)})
$$

### Batch gradients

Averaging across all samples:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{2}{n} \sum_{i=1}^{n} x^{(i)\mathsf{T}} (\hat{y}^{(i)} - y^{(i)}) = \frac{2}{n} X^{\mathsf{T}} (\hat{Y} - Y)
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{2}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)}) = \frac{2}{n} \mathbf{1}^{\mathsf{T}} (\hat{Y} - Y)
$$

where $X \in \mathbb{R}^{n \times d}$, $\hat{Y}, Y \in \mathbb{R}^{n \times 1}$, and $\mathbf{1} \in \mathbb{R}^{n \times 1}$ is a column vector of ones.

## Gradient Descent Update

With learning rate $\eta$:

$$
W \leftarrow W - \eta \frac{\partial \mathcal{L}}{\partial W}
$$

$$
b \leftarrow b - \eta \frac{\partial \mathcal{L}}{\partial b}
$$

## Implementation

Let's implement the model from scratch using Python and NumPy.

```python
%matplotlib inline
import numpy as np

class MyOwnLinearRegression:
    def __init__(self, learning_rate=0.0001, n_iters=30000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient Descent
        for _ in range(self.n_iters):
            # Predict the target values
            y_predicted = np.dot(X, self.weights) + self.bias

            # Compute gradients
            dw = (2 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (2 / n_samples) * np.sum(y_predicted - y)

            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias


import matplotlib.pyplot as plt
import pandas as pd

# Load dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Example usage
regressor = MyOwnLinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)

# Visualize the results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```
