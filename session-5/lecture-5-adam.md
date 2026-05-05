#  Adam (Adaptive Moment Estimation)

![](./img/9.gif)

---

## 1. Motivation

From the previous lecture, **momentum** accelerates learning along consistent directions and reduces oscillations.

However, there is another issue:

* Different parameters can have **very different gradient scales**
* Using a single global learning rate may be inefficient

We need an optimizer that combines:

1. **Directional memory** (momentum)
2. **Parameter-specific adaptive step sizes**

This leads to **Adam (Adaptive Moment Estimation)**.

---

## 2. Review — Mini-Batch Gradient Descent

Mini-batch SGD updates parameters as:

$$
 g = \frac{1}{B} \sum_{i \in \mathcal{B}} \frac{\partial \mathcal{L}_i}{\partial W}, \quad W \leftarrow W - \eta g
$$

Momentum adds a **velocity term** $v$ that smooths $g$. Adam goes further by also adapting the **learning rate per parameter**.

---

## 3. Adam — First and Second Moments

![](./img/5.gif)


Adam tracks two moving averages:

### 3.1 First Moment — Mean of Gradients

$$
m^{(t)} = \beta_1 m^{(t-1)} + (1-\beta_1) g
$$

* Similar to momentum's $v$: both smooth the raw gradient $g$
* Captures **directional trend** of gradients
* $\beta_1$ typically 0.9

> [!NOTE]
> **Why $m$ instead of $v$?**
> In the momentum lecture we called the smoothed gradient $v$ (velocity). Adam uses $m$ (moment) for the same idea. Adam's $v$ is a *different* quantity — it tracks the volatility (squared gradients).

---

### 3.2 Second Moment — Mean of Squared Gradients

$$
v^{(t)} = \beta_2 v^{(t-1)} + (1-\beta_2) g^2
$$

* Measures **gradient magnitude and variability**
* **Represents the "Volatility" of the gradients:**
    * Large $v$ → **High volatility** (unstable/steep) → reduce step size for safety.
    * Small $v$ → **Low volatility** (stable/flat) → increase step size to speed up.
* $\beta_2$ typically 0.999 (long-term memory of volatility)

---


## 4. Bias Correction

Moving averages ($m^{(t)}, v^{(t)}$) are initialized at **0**. Since $\beta_1$ and $\beta_2$ are typically very close to 1 (e.g., 0.999), the moving averages are heavily "dragged" toward zero during the first few iterations.

To fix this, we compute **bias-corrected** moments:

$$
\hat{m}^{(t)} = \frac{m^{(t)}}{1-(\beta_1)^t}
$$

$$
\hat{v}^{(t)} = \frac{v^{(t)}}{1-(\beta_2)^t}
$$

> [!CAUTION]
> **Notation Alert:** 
> In the expressions above, **$t$ is an exponent**, not an index. 
> *   $\beta_1, \beta_2$: Constant hyperparameters (e.g., 0.9, 0.999).
> *   $(\beta)^t$: The value of $\beta$ raised to the **power of the current time step $t$**.

### Why is this necessary?
*   **The "Cold Start" Problem:** Without correction, $m^{(1)}$ would be $(1-\beta_1)g$. If $\beta_1 = 0.9$, your first update is only **10%** of what it should be.
*   **The Fix:** At $t=1$, the denominator $1-(0.9)^1 = 0.1$. Dividing by $0.1$ effectively scales the moment back up to its true magnitude.
*   **Self-Vanishing:** As training progresses ($t \to \infty$), the term $(\beta)^t$ quickly approaches **0**. The correction factor $1/(1-\beta^t)$ becomes **1**, meaning the correction naturally fades away once the moving averages become stable.

**Key Outcome:** Ensures **early updates are not underestimated**, preventing the model from getting "stuck" or moving too slowly during the first few batches.

---

## 5. Adam Update Rule

Parameter update:

$$
W \leftarrow W - \eta \frac{\hat{m}}{\sqrt{\hat{v}} + \epsilon}
$$

Where:

* $W$ — parameters
* $\eta$ — base learning rate (typical 0.001)
* $\hat{m}$ — bias-corrected first moment
* $\hat{v}$ — bias-corrected second moment
* $\epsilon$ — small number for numerical stability (e.g., $10^{-8}$)

**Key idea:** Step size is **adapted per parameter** based on gradient history.

---

## 6. Geometric Intuition

![](./img/8.gif)


1. **First moment $m$** → smooths noisy gradients (like momentum)
2. **Second moment $v$** → scales updates according to gradient volatility
3. **Combined effect** → move quickly along flat, stable directions and cautiously along steep, noisy directions

Analogy: A ship navigating in fog:

* Trust **consistent directions** from history → first moment
* Reduce trust in **unreliable directions** → second moment

---

## 7. Practical Defaults

| Hyperparameter       | Typical Value |
| -------------------- | ------------- |
| Learning rate $\eta$ | 0.001         |
| $\beta_1$            | 0.9           |
| $\beta_2$            | 0.999         |
| $\epsilon$           | 1e-8          |

* Works well for most deep learning tasks
* Minimal tuning required

---

## 8. Why Adam Works Well

* Combines **SGD efficiency** with **momentum smoothing**
* Adapts **step size per parameter**
* Handles **noisy gradients**, **saddle points**, and **high-curvature landscapes**
* Converges quickly and robustly

**Takeaway:** Adam is a **safe, powerful default optimizer** for deep networks.

---

## 9. PyTorch Example

```python
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    betas=(0.9, 0.999), # Optional: defaults are fine
    eps=1e-8 # Optional: defaults are fine
)

for X, y in dataloader:
    optimizer.zero_grad()
    prediction = model(X)
    loss = criterion(prediction, y)
    loss.backward()
    optimizer.step()
```

---

## 10. Summary

Adam is the natural culmination of **mini-batch SGD + momentum + adaptive learning rates**:

1. Tracks **first moment** → smooths gradients (direction)
2. Tracks **second moment** → scales steps (magnitude)
3. Bias correction → stabilizes early training

> Adam allows neural networks to **learn efficiently and robustly** across complex, high-dimensional loss landscapes.
