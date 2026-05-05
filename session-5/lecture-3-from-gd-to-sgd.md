# From Gradient Descent to Stochastic Gradient Descent

![](./img/s1.jpg)


---

## 1. Full-Batch Gradient Descent

Recall standard gradient descent on the full dataset. The gradient is the average over all $n$ samples:

$$
g = \frac{\partial \mathcal{L}}{\partial W} = \frac{1}{n} \sum_{i=1}^n \frac{\partial \mathcal{L}_i}{\partial W}, \quad W \leftarrow W - \eta g
$$

**Observation:** Computing $g$ requires summing over all $n$ samples each step.

* Accurate gradient estimates
* Smooth convergence
* But very expensive for large datasets

---

## 2. The Inefficiency Problem

For modern datasets:

* $n$ can be millions or more
* Each iteration takes a long time
* Full-batch updates are slow, even if precise

**Key insight:** Exact gradient is not always necessary — an approximate gradient is often sufficient for learning.

---

## 3. Stochastic Gradient Descent (SGD) & Mini-batch SGD

**Stochastic Gradient Descent** in its **theoretical definition** uses only **one sample** per step (Batch Size $B = 1$):

$$
g = \frac{\partial \mathcal{L}_i}{\partial W}, \quad W \leftarrow W - \eta g
$$

**But in modern deep learning practice**, the term **"SGD" almost always refers to "Mini-batch SGD"** — using a **small random subset** to approximate the gradient:

$$
g = \frac{1}{B} \sum_{i \in \mathcal{B}} \frac{\partial \mathcal{L}_i}{\partial W}, \quad W \leftarrow W - \eta g
$$

Where:

* $B$ — **Batch size** ($1 < B \ll n$)
* $\mathcal{B}$ — The set of indices for the current mini-batch
* $g$ — The mini-batch gradient

### Important Terminology Note:

Depending on the batch size $B$, the algorithm is named differently:

| Name | Batch Size ($B$) | Description |
| :--- | :--- | :--- |
| **Batch GD** | $B = n$ | Uses every sample. Accurate but very slow. |
| **One-sample SGD** | $B = 1$ | Uses only **one** sample per step. Extremely noisy. |
| **Mini-batch SGD**| $1 < B < n$ | The "Goldilocks" zone. Fast, stable, and hardware-efficient. |

> In this course and all subsequent discussions, unless otherwise specified, **"SGD" refers to "Mini-batch SGD"**. If we mean the one-sample version, we will explicitly say "One-sample SGD" or "Single-sample SGD".

**Advantages:**

* Updates are cheaper than full-batch GD
* Faster per iteration
* Introduces stochasticity that can help optimization

---

## 4. Intuition: Noisy Descent Helps

![](./img/s2.jpg)


* Mini-batch updates are **noisy approximations** of the true gradient
* Noise allows escaping **shallow local minima or saddle points**

**Visualization:** Descending a bumpy valley:

* Full-batch GD → precise but slow path
* Mini-batch SGD → jittery path that explores valleys and escapes plateaus

![](./img/saddlepoint.jpg)

---

## 5. Choosing Batch Size

* Small batch → more noise, faster iteration
* Large batch → smoother gradient, slower iteration

Typical guidelines:

| Dataset Size | Batch Size |
| ------------ | ---------- |
| Small        | 16–64      |
| Medium       | 64–256     |
| Large        | 256–1024+  |

> Batch size interacts with learning rate: smaller batches often require smaller $\eta$ for stability.

---

## 6. PyTorch Example

```python
import torch
from torch.utils.data import DataLoader

# DataLoader automatically creates mini-batches
# This is the standard Mini-batch SGD practice
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

# torch.optim.SGD works with mini-batches by default
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

for epoch in range(num_epochs):
    for X, y in dataloader:  # Each loop: one mini-batch
        optimizer.zero_grad()
        prediction = model(X)
        loss = criterion(prediction, y)
        loss.backward()            # Compute g on this mini-batch
        optimizer.step()           # W <- W - eta * g
```

---

## 7. Summary

* Full-batch GD: precise but computationally expensive
* Mini-batch SGD: cheaper, faster, introduces helpful noise
* Batch size controls the trade-off between gradient variance and efficiency

**Key takeaway:** Modern deep learning almost always relies on **Mini-batch SGD** (commonly just called "SGD").