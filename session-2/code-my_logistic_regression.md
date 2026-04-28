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


### The Mathematics of Gradient Descent for Logistic Regression

Adopting a row-vector convention, a single training sample is represented as:

\[
\mathbf{x}^{(i)} \in \mathbb{R}^{1 \times d}.
\]

The Logistic Regression model first computes a linear score:

\[
z^{(i)} = \mathbf{x}^{(i)} \mathbf{W} + b,
\]
where
\[
\mathbf{x}^{(i)} \in \mathbb{R}^{1 \times d},\quad \mathbf{W} \in \mathbb{R}^{d \times 1},\quad b \in \mathbb{R}.
\]
Consequently, \( z^{(i)} \in \mathbb{R} \) is a scalar.

This score is passed through the sigmoid function to obtain a predicted probability for class 1:
\[
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{so} \quad \hat{y}^{(i)} = \sigma(z^{(i)}).
\]

The model's performance is measured using the **Binary Cross-Entropy Loss** averaged over all \( m \) samples:
\[
\mathcal{L}(\mathbf{W}, b) = -\frac{1}{m} \sum_{i=1}^{m} \Big[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big].
\]

To perform gradient descent, we require the gradients:
\[
d\mathbf{W} = \frac{\partial \mathcal{L}}{\partial \mathbf{W}} \quad \text{and} \quad db = \frac{\partial \mathcal{L}}{\partial b}.
\]

#### Gradient Derivation for a Single Sample
For an individual sample \( i \), the loss is:
\[
\ell^{(i)} = -\Big[ y^{(i)} \log \hat{y}^{(i)} + (1 - y^{(i)}) \log(1 - \hat{y}^{(i)}) \Big].
\]

Applying the chain rule through the computational graph \( \mathbf{W} \to z^{(i)} \to \hat{y}^{(i)} \to \ell^{(i)} \):
\[
\frac{\partial \ell^{(i)}}{\partial \mathbf{W}} = \frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial \mathbf{W}}.
\]

1.  **Gradient of the loss w.r.t. the prediction:**
    \[
    \frac{\partial \ell^{(i)}}{\partial \hat{y}^{(i)}} = -\Bigg( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \Bigg).
    \]

2.  **Gradient of the sigmoid activation:**
    \[
    \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1 - \hat{y}^{(i)}).
    \]

3.  **Gradient of the linear score w.r.t. the weights:**
    \[
    \frac{\partial z^{(i)}}{\partial \mathbf{W}} = \mathbf{x}^{(i)\top}.
    \]

Substituting these terms:
\[
\frac{\partial \ell^{(i)}}{\partial \mathbf{W}} = -\Bigg( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \Bigg) \cdot \hat{y}^{(i)}(1 - \hat{y}^{(i)}) \cdot \mathbf{x}^{(i)\top}.
\]

A remarkable simplification occurs:
\[
\frac{\partial \ell^{(i)}}{\partial \mathbf{W}} = (\hat{y}^{(i)} - y^{(i)}) \cdot \mathbf{x}^{(i)\top}.
\]

#### Averaging Over the Entire Dataset
Let \( \mathbf{X} \in \mathbb{R}^{m \times d} \) be the data matrix (stacked row vectors \( \mathbf{x}^{(i)} \)), and \( \mathbf{y}, \hat{\mathbf{y}} \in \mathbb{R}^{m \times 1} \) be the stacked labels and predictions. Averaging the gradient for the weights gives:
\[
\frac{\partial \mathcal{L}}{\partial \mathbf{W}} = \frac{1}{m} \mathbf{X}^{\top} (\hat{\mathbf{y}} - \mathbf{y}).
\]

For the bias term, since \( \frac{\partial z^{(i)}}{\partial b} = 1 \), the derivation yields:
\[
\frac{\partial \ell^{(i)}}{\partial b} = \hat{y}^{(i)} - y^{(i)}.
\]
Averaging over the dataset:
\[
\frac{\partial \mathcal{L}}{\partial b} = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}).
\]

#### The Gradient Descent Update Rule
The parameters are then updated with a learning rate \( \eta \):
\[
\mathbf{W} := \mathbf{W} - \eta \cdot d\mathbf{W}, \quad b := b - \eta \cdot db.
\]

**Key Insight:** Despite the nonlinear sigmoid transformation, the final gradient for logistic regression with cross-entropy loss simplifies elegantly to \( (\hat{\mathbf{y}} - \mathbf{y}) \), analogous to linear regression. This makes the gradient computation both clean and efficient.


### Why Use Cross-Entropy Loss Instead of Mean Squared Error (MSE)?

Notice the final gradient formulas:
\[
d\mathbf{W} = \frac{1}{m} \mathbf{X}^{\top} (\hat{\mathbf{y}} - \mathbf{y}), \quad db = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}^{(i)} - y^{(i)}).
\]
These bear a striking, **beautiful** resemblance to the gradients in linear regression. This simplicity is a direct consequence of pairing the cross-entropy loss with the sigmoid activation.

**Primary advantages of cross-entropy over MSE for logistic regression:**

1.  **Convexity:** The loss function is convex, guaranteeing a single global minimum and more reliable optimization.
2.  **Gradient Scaling:** It penalizes confident but wrong predictions much more heavily, providing stronger, more informative gradients to drive learning, especially when predictions are very incorrect.
3.  **Mitigates Vanishing Gradients:** For extreme predictions (e.g., \( \hat{y} \approx 0 \) or \( 1 \)), the gradient of the cross-entropy loss with respect to the inputs remains significant. In contrast, MSE gradients can become extremely small in these "flat" saturation regions of the sigmoid, drastically slowing down learning.

This combination—sigmoid activation for probability interpretation and cross-entropy loss for effective gradient propagation—forms the mathematically coherent and computationally efficient foundation for binary classification.


## Testing the Gradient Descent Implementation

Let's test our implementation on a simple classification dataset:

```python
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate a simple classification dataset
X, y = datasets.make_classification(
    n_samples=100, n_features=2, n_redundant=0, 
    n_informative=2, random_state=1, n_clusters_per_class=1
)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create and train the logistic regression model
model = MyOwnLogisticRegressionGD(learning_rate=0.01, n_iters=1000)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Visualize the results
def plot_decision_boundary(X, y, model):
    # Define the bounds of the plot
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Create a mesh grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    
    # Make predictions on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # Create a contour plot
    plt.contourf(xx, yy, Z, alpha=0.3)
    
    # Plot the training points
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', s=50)
    plt.title('Logistic Regression Decision Boundary')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot the decision boundary
plot_decision_boundary(X, y, model)
```

You should be able to see the decision boundary is a line, and has math formulation:

$$w_1 x_1 + w_2 x_2 + b = 0$$

which is equivalent to:

$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$
