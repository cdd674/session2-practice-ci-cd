

# 1. Left-Side vs. Right-Side Projection

There are **two equally correct but fundamentally different coordinate systems** for thinking about linear transformations.

---

## 1.1 Classical Linear Algebra (Column-Vector World)

The canonical definition:

$$
y = W x
$$

* $x \in \mathbb{R}^{d_{in} \times 1}$
* $W \in \mathbb{R}^{d_{out} \times d_{in}}$
* $y \in \mathbb{R}^{d_{out} \times 1}$

### Interpretation

* The matrix **acts on the vector from the left**
* The vector is an **object being transformed**

This is the dominant view in:

* mathematics
* physics
* control theory

---

## 1.2 Modern ML Systems (Row-Vector World)

The dominant formulation:

$$
y = x W
$$

* $x \in \mathbb{R}^{1 \times d_{in}}$
* $W \in \mathbb{R}^{d_{in} \times d_{out}}$
* $y \in \mathbb{R}^{1 \times d_{out}}$

### Interpretation

* The data **flows through the matrix**
* The matrix is a **projection layer**

This is the dominant view in:

* NumPy
* PyTorch
* TensorFlow
* sklearn

---

## 1.3 Key Equivalence

These two are mathematically identical:

$$
y = W x \quad \Longleftrightarrow \quad y^T = x^T W^T
$$

So:

> **Right-multiplication is just the transpose-view of left-multiplication**

But computationally, they behave very differently.

---

# 2. Why ML Uses Right-Multiplication

This is not arbitrary. It is driven by **systems constraints**.

---

## 2.1 Data is Stored as Rows

In ML:

$$
X \in \mathbb{R}^{n_{samples} \times d_{features}}
$$

Each row = one sample.

So:

$$
Y = X W
$$

* processes all samples at once
* no reshaping needed

---

## 2.2 Batch Computation Becomes Natural

Right-multiplication scales cleanly:

$$
X W
$$

* $X$: $(n, d_{in})$
* $W$: $(d_{in}, d_{out})$
* $Y$: $(n, d_{out})$

This gives:

* vectorization
* SIMD efficiency
* GPU acceleration

---

## 2.3 Memory Layout (Row-Major)

NumPy and PyTorch use **row-major memory layout**.

This means:

* rows are contiguous in memory
* iterating over rows is cache-friendly

Right-multiplication aligns with this layout.

---

## 2.4 Functional Composition Looks Like Pipelines

Deep learning layers:

$$
X \rightarrow XW_1 \rightarrow XW_1W_2 \rightarrow \cdots
$$

This reads like a **data pipeline**, not operator application.

---

# 3. NumPy: The Ground Truth Implementation

NumPy reveals the mechanics clearly.

---

## 3.1 Standard Right-Multiplication

```python
import numpy as np

X = np.random.randn(5, 4)   # 5 samples, 4 features
W = np.random.randn(4, 3)   # projection to 3 features

Y = X @ W                   # (5, 3)
```

### Interpretation

Each row:

$$
y_i = x_i W
$$

---

## 3.2 Left-Multiplication Exists but Feels Awkward

```python
x = np.random.randn(4, 1)
W = np.random.randn(3, 4)

y = W @ x   # (3, 1)
```

Problems:

* batching is unnatural
* stacking requires column concatenation

---

## 3.3 The 1D Vector Trap

```python
x = np.random.randn(4)
A = np.random.randn(4, 3)

A @ x   # OK
x @ A   # also OK (but ambiguous)
```

This works because NumPy treats `(4,)` as:

* sometimes column
* sometimes row

This is **not true linear algebra**, but a convenience rule.

---

## 3.4 Best Practice

Always enforce 2D:

```python
x = x.reshape(1, -1)   # row vector
```

---

# 4. sklearn: Right-Multiplication Hidden in APIs

sklearn enforces a strict convention:

$$
X \in \mathbb{R}^{n \times d}
$$

---

## 4.1 Linear Models

Internally:

$$
\hat{y} = X w + b
$$

Each prediction:

$$
\hat{y}_i = x_i w
$$

---

## 4.2 PCA Example


```python
from sklearn.decomposition import PCA

# X: (n_samples, n_features)
X = np.random.randn(100, 4) 
pca = PCA(n_components=2)
pca.fit(X)

# Internally: X_transformed = X @ pca.components_.T
# The components_ are stored as (n_components, n_features), 
# so sklearn transposes them to align for a RIGHT-MULTIPLY.
X_reduced = pca.transform(X) # Result Shape: (100, 2)
```


---

## 4.3 Design Philosophy

sklearn enforces:

> rows = samples, always

This removes ambiguity entirely.

---

# 5. PyTorch: Right-Multiplication with Internal Transpose

PyTorch formalizes this pattern.

---

## 5.1 Linear Layer

Definition:

```python
nn.Linear(in_features=4, out_features=2)
```

Internally:

* weight shape: $(2, 4)$

But forward pass:

$$
y = x W^T + b
$$

---

## 5.2 Why Store Transposed?

Because:

* gradient efficiency

---

## 5.3 Batch Computation

```python
x = torch.randn(32, 4)
layer = nn.Linear(4, 2)

y = layer(x)   # (32, 2)
```

Equivalent to:

$$
Y = X W^T + b
$$

---

## 5.4 Broadcasting

```python
A = torch.randn(10, 3, 4)
x = torch.randn(4)

Y = A @ x   # (10, 3)
```

This extends right-multiplication to higher dimensions.

---

# 6. Deep Insight: Rows vs Columns

This is the most important conceptual bridge.

---

## 6.1 Right-Multiplication = Row-wise Dot Products

$$
Y = X W
$$

means:

$$
y_i = x_i W
$$

Each row is processed independently.

---

## 6.2 Left-Multiplication = Column Transformation

$$
y = W x
$$

means:

* mixing input dimensions
* transforming basis

---

## 6.3 Dual Interpretation

| Perspective | Meaning                  |
| ----------- | ------------------------ |
| $Wx$        | transform vector         |
| $xW$        | transform representation |

---

# 7. Gradient Flow Differences

This is where ML interpretation becomes critical.

---

## 7.1 Right-Multiplication Backprop

Forward:

$$
Y = X W
$$

Gradients:

$$
\frac{\partial L}{\partial X} = \frac{\partial L}{\partial Y} W^T
$$

$$
\frac{\partial L}{\partial W} = X^T \frac{\partial L}{\partial Y}
$$

---

## 7.2 Interpretation

* gradients flow **backwards through transpose**
* weight update uses **outer products of data and gradients**

---

## 7.3 Why This Matters

Right-multiplication:

* aligns with minibatch gradients
* produces efficient matrix operations

---

# 9. Unified View: Projection vs Aggregation

This resolves the confusion completely.

---

## 9.1 Right-Multiplication

$$
X W
$$

= **projection**

* map features
* apply transformation
* forward data

---

## 9.2 Left-Multiplication

$$
\alpha^T V
$$

= **aggregation**

* combine rows
* weighted sum
* reduce information

---

## 9.3 Final Principle

> Right-multiplication transforms representations
> Left-multiplication aggregates information

---

# 10. Final Comparison Table

| Aspect           | Left ($W x$)   | Right ($x W$) |
| ---------------- | -------------- | ------------- |
| Data format      | column vector  | row vector    |
| Typical domain   | math, physics  | ML, DL        |
| Batch support    | difficult      | natural       |
| Interpretation   | operator acts  | data flows    |
| Role             | transformation | projection    |

---

# 11. Final Mental Model

A clean way to think about everything:

* **Classical math**: vectors are objects, matrices act on them
* **Machine learning**: data is a stream, matrices are layers

So:

$$
\text{ML pipeline} = X W_1 W_2 W_3 \cdots
$$

not:

$$
W_3 W_2 W_1 x
$$

---

# 12. One Sentence Summary

> Left-multiplication is about how operators act on vectors.
> Right-multiplication is about how data flows through systems.
