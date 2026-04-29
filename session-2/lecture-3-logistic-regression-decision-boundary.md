# Decision Boundary

---

## 1. From probability to decision


![](./img/lg1.jpg)



The model outputs:

$$
\hat{y} = \sigma(x W + b)
$$

where $x \in \mathbb{R}^{1 \times d}$ is a row vector, $W \in \mathbb{R}^{d \times 1}$ is a weight matrix, and $b \in \mathbb{R}^{1 \times 1}$ is the bias (scalar as row vector).

This is a probability.

But classification requires a decision:

$$
y \in {0, 1}
$$

So we need a rule to convert probability into a label.

---

## 2. The decision rule

![](./img/decisionboundary.jpg)

A natural choice:

$$
\hat{y} > 0.5 \Rightarrow 1
$$

$$
\hat{y} \leq 0.5 \Rightarrow 0
$$

This uses 0.5 as the threshold.

---

## 3. Equivalent form

Recall:

$$
\sigma(0) = 0.5
$$

So:

$$
\hat{y} > 0.5
\quad \Longleftrightarrow \quad
x W + b > 0
$$

Therefore the decision rule becomes:

$$
x W + b > 0 \Rightarrow 1
$$

$$
x W + b \leq 0 \Rightarrow 0
$$

---

## 4. The decision boundary



![](./img/decisionboundary2.jpg)


The boundary occurs when the model is uncertain:

$$
\hat{y} = 0.5
$$

This corresponds to:

$$
x W + b = 0
$$

This equation defines the decision boundary.

#### In 2-D settings:

The decision boundary is a line, and has math formulation:

$$w_1 x_1 + w_2 x_2 + b = 0$$

which is equivalent to (back to high school/middle school algebra):

$$x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$


---

## 5. Important insight


![](./img/logisticregressionanimatedgif4.gif)



Even though the model uses a nonlinear function:

$$
\boxed{\sigma(x W + b)}
$$

the decision boundary is still:

$$
\boxed{x W + b = 0}
$$

which is linear.
