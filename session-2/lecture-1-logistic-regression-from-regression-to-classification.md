# From Regression to Classification

![](./img/lg2.jpg)

---

## 1. What did linear regression do?

Linear regression models a real-valued output:

$$
\hat{y} = x W + b
$$

where $x \in \mathbb{R}^{1 \times d}$ is a row vector, $W \in \mathbb{R}^{d \times 1}$ is a weight matrix, and $b \in \mathbb{R}^{1 \times 1}$ is the bias (scalar as row vector).

It answers:

> Given input $x$, what number should we predict?

---

## 2. When does this break?

Now consider a classification problem.

Example:

* Email → spam or not spam
* Tumor → benign or malignant

The label is:

$$
y \in \{0, 1\}
$$

But linear regression gives:

$$
\hat{y} \in (-\infty, +\infty)
$$

This creates a mismatch.

---

## 3. Why is this a problem?

### Problem 1: Invalid outputs

A prediction like:

$$
\hat{y} = 3.7
$$

has no meaning for classification.

---

### Problem 2: No notion of confidence

We want:

* How confident is the model?
* How likely is class 1?

Linear regression does not answer this.

---

### Problem 3: Thresholding is unstable

We could try:

$$
\hat{y} > 0.5 \Rightarrow 1
$$

But this is problematic:

* Outputs are unbounded
* Small changes in $w$ can produce large shifts in $\hat{y}$
* No guarantee of consistent behavior

---

## 4. What do we actually want?

Not a number.

We want a probability:

$$
P(y = 1 \mid x)
$$

This gives:

* A value between 0 and 1
* A measure of confidence
* A principled decision rule

---

## 5. The core transformation

We still like the linear structure:

$$
z = x W + b
$$

where $x \in \mathbb{R}^{1 \times d}$ is a row vector, $W \in \mathbb{R}^{d \times 1}$ is a weight matrix, and $b \in \mathbb{R}^{1 \times 1}$ is the bias (scalar as row vector).

But we need to transform:

$$
z \in (-\infty, +\infty)
\quad \longrightarrow \quad
\hat{y} \in [0, 1]
$$

This is the key problem.

---

## 6. What kind of function do we need?

We need a function that:

* Takes any real number
* Outputs a value between 0 and 1
* Is smooth and differentiable
* Preserves ordering (Monotonic)

In other words:

$$
f: \mathbb{R} \rightarrow [0, 1]
$$


![](./img/sigmoid3b1b.jpg)