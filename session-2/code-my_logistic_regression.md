# Logistic Regression from Scratch

## From Linear Regression to Logistic Regression

To understand Logistic Regression, it's helpful to first recap Linear Regression. In previous session, we implemented Linear Regression from scratch:


```python
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
```

The key difference between Linear Regression and Logistic Regression is that Logistic Regression applies a sigmoid function to the linear model output to transform the predictions into probabilities between 0 and 1. This makes Logistic Regression suitable for binary classification problems where we need to predict one of two possible outcomes.


## Implementation: Logistic Regression with Gradient Descent

Let's implement Logistic Regression using Gradient Descent:

```python
import numpy as np

class MyOwnLogisticRegressionGD:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iters):
            # Calculate linear model
            linear_model = np.dot(X, self.weights) + self.bias
            # Apply sigmoid function
            y_predicted = self._sigmoid(linear_model)

            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return np.array(y_predicted_cls)

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
```






### The Math Behind Gradient Descent for Logistic Regression

Under our row-vector convention, each training sample is written as:

$$
x^{(i)} \in \mathbb{R}^{1 \times d}
$$

The Logistic Regression model first computes a linear projection:

$$
z^{(i)} = x^{(i)} W + b
$$

where:

- $x^{(i)} \in \mathbb{R}^{1 \times d}$
- $W \in \mathbb{R}^{d \times 1}$
- $b \in \mathbb{R}^{1 \times 1}$

Thus:

$$
z^{(i)} \in \mathbb{R}^{1 \times 1}
$$

This scalar output is then transformed into a probability using the sigmoid function:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

So:

$$
\hat{y}^{(i)} = \sigma(z^{(i)})
$$

where $\hat{y}^{(i)}$ is the predicted probability that sample $i$ belongs to class 1.

To measure how well predictions match true labels, we use Binary Cross-Entropy Loss over all $n$ samples:

$$
\mathcal{L}(W, b) = -\frac{1}{n} \sum_{i=1}^{n} \left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

To perform Gradient Descent, we need:

$$
dW = \frac{\partial \mathcal{L}}{\partial W}
\quad \text{and} \quad
db = \frac{\partial \mathcal{L}}{\partial b}
$$

For one sample, define:

$$
\ell^{(i)} = -\left[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \right]
$$

Because the loss depends on $W$ through:

$$
W \rightarrow z^{(i)} \rightarrow \hat{y}^{(i)} \rightarrow \ell^{(i)}
$$

we apply the chain rule:

$$
\frac{\partial \ell^{(i)}}{\partial W} = \frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial W}
$$

**First:**

$$
\frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} = -\left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right)
$$

**Second**, since the sigmoid derivative is:

$$
\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1 - \hat{y}^{(i)})
$$

**Third**, because:

$$
z^{(i)} = x^{(i)}W + b
$$

we get:

$$
\frac{\partial z^{(i)}}{\partial W} = x^{(i)\mathsf{T}}
$$

**Substituting:**

$$
\frac{\partial \ell^{(i)}}{\partial W} = -\left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right) \cdot \hat{y}^{(i)}(1 - \hat{y}^{(i)}) \cdot x^{(i)\mathsf{T}}
$$

After simplification, the complex terms collapse beautifully into:

$$
\frac{\partial \ell^{(i)}}{\partial W} = x^{(i)\mathsf{T}}(\hat{y}^{(i)} - y^{(i)})
$$

Averaging across all samples:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^{\mathsf{T}}(\hat{Y} - Y)
$$

For the bias term, since:

$$
\frac{\partial z^{(i)}}{\partial b} = 1
$$

we similarly obtain:

$$
\frac{\partial \ell^{(i)}}{\partial b} = \hat{y}^{(i)} - y^{(i)}
$$

and over the full dataset:

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})
$$

This gives the final Gradient Descent updates:

$$
W \leftarrow W - \eta \, dW
$$

$$
b \leftarrow b - \eta \, db
$$

The key insight is that although Logistic Regression combines:

- linear projection
- sigmoid nonlinearity
- logarithmic loss

the final gradient simplifies to:

$$
\hat{y} - y
$$

This elegant simplification is why Sigmoid + Binary Cross-Entropy is such a powerful pairing: mathematically clean, computationally efficient, and ideal for binary classification.



You'll notice that for the derivatives $\frac{\partial \mathcal{L}}{\partial W}$ and $\frac{\partial \mathcal{L}}{\partial b}$ we have:

$$
\frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} X^{\mathsf{T}} \cdot (\hat{Y} - Y)
$$

$$
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}^{(i)} - y^{(i)})
$$

These formulas look remarkably similar to the ones used for Linear Regression (1/n for logistic regression, 2 for linear regression), which is a **beautiful** result of using cross-entropy loss with the sigmoid activation function.

