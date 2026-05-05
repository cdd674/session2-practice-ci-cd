# From Gradient to Parameter Update

![](./img/0.gif)

---

## 1. Review — What We Already Have

By now, after backpropagation, we can compute the gradient of the loss $\mathcal{L}$ with respect to the parameters $W$:

$$
g = \frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathcal{L}_i}{\partial W}
$$

We denote this gradient compactly by $g$. This tells us **the direction in which the loss increases**.

---

## 2. The Key Question

```text
What do we do with this gradient?
```

* Computing $\frac{\partial \mathcal{L}}{\partial W}$ gives information, but **no actual learning happens yet**.
* Gradient alone does not update the model.

---

## 3. The Core Idea of Optimization

The basic principle of optimization in deep learning:

```text
Move parameters in the direction that reduces the loss.
```

Formally, we define an **update rule** to translate gradient information into parameter changes.

---

## 4. Gradient Descent Update Rule

![](./img/3.gif)


The simplest update rule is **gradient descent (GD)**:

$$
W \leftarrow W - \eta g, \quad g = \frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathcal{L}_i}{\partial W}
$$

> This is the most fundamental bridge from "having a gradient" to "actually learning".

---

## 5. Intuition Behind Gradient Descent

* Gradient is the **slope** of the loss function along the parameter dimension.
* Negative gradient points toward **steepest descent**.
* By moving a small step $\eta$ in that direction, we reduce the loss:

```text
Gradient = slope
Move downhill = reduce loss
```

---

## 6. Key Insight — Gradient ≠ Learning

Computing gradients is only **information gathering**.
**Learning happens only when we apply updates**:

$$
W \leftarrow W - \eta g
$$

* Gradient alone tells us **where to go**.
* Update rule tells us **how far to go**.
* Step size $\eta$ controls **speed** of learning.

---

## 7. PyTorch Example

```python
import torch
import torch.nn as nn

# A simple linear layer: z = xW + b
model = nn.Linear(d_in, d_out)

# Loss function
criterion = nn.MSELoss()

# Optimizer: applies W <- W - eta * g
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for X, y in dataloader:
    optimizer.zero_grad()      # Clear old gradients
    prediction = model(X)      # Forward pass
    loss = criterion(prediction, y)
    loss.backward()            # Compute g = dL/dW
    optimizer.step()             # Apply W <- W - eta * g
```

---

## 8. Summary

1. Gradient gives **direction**, not learning.
2. Update rule translates gradient into **actual parameter changes**.
3. Gradient descent is the simplest **optimization rule**:

$$
W \leftarrow W - \eta g
$$
