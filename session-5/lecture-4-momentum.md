#  Momentum


![](./img/7.gif)



---

## 1. Review — Mini-Batch Gradient Descent

From the previous lecture, mini-batch gradient descent (SGD) updates parameters as:

$$
W \leftarrow W - \eta g, \quad g = \frac{1}{B} \sum_{i \in \mathcal{B}} \frac{\partial \mathcal{L}_i}{\partial W}
$$

Mini-batch SGD is efficient for large datasets, but the stochasticity (noise) in each mini-batch gradient can cause oscillations in **narrow valleys** or **high-curvature directions**.

**Problem:** Even with a good learning rate, SGD can be **slow to converge** in some directions due to oscillations.

---

## 2. Motivation for Momentum


![](./img/1.gif)


Imagine descending a **long, narrow valley**:

* Gradient along steep direction → large updates → zig-zag path
* Gradient along shallow direction → small updates → slow progress

We want:

1. **Reduce oscillations** along steep directions
2. **Accelerate convergence** along shallow directions

**Solution:** Momentum — treat updates like a **moving ball** that accumulates velocity.

---

## 3. Momentum Update Rule

Momentum introduces a **velocity term** $v$ that smooths the gradient $g$ over time:

$$
v \leftarrow \beta v + (1 - \beta) g
$$

$$
W \leftarrow W - \eta v
$$

In row-vector notation for a linear layer:
$$
z = xW + b, \quad v \leftarrow \beta v + (1-\beta) g, \quad W \leftarrow W - \eta v
$$

Where:

* $g$ — mini-batch gradient
* $v$ — accumulated velocity (smoothed gradient)
* $\beta \in [0,1)$ — momentum coefficient (commonly 0.9)
* $\eta$ — learning rate

**Interpretation:**

* $v$ smooths the gradient over multiple steps
* $\beta$ controls how much past gradients influence the current update
* The parameter moves **faster in consistent directions** and **slows down in oscillating directions**

---

## 4. Geometric Intuition

* Without momentum → zig-zag path along steep walls
* With momentum → smooth path, like a ball rolling down the valley

Visual analogy:

* Steep, high-curvature direction → velocity cancels oscillations
* Flat, shallow direction → velocity accumulates, accelerates learning

**Key insight:** Momentum balances **stability** and **speed**.

---

## 5. Practical Tips

1. Typical momentum coefficient: $\beta = 0.9$
2. Combine with mini-batch SGD for best results
3. Works especially well for **deep networks** with **high-curvature loss surfaces**
4. Learning rate may need slight adjustment when using momentum

**Summary:** Momentum lets parameters **carry speed in consistent directions**, reducing oscillation and accelerating learning.

---

## 6. PyTorch Example

```python
import torch.optim as optim

# SGD with momentum: v <- beta * v + (1-beta) * g, then W <- W - eta * v
optimizer = optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9  # beta = 0.9
)

for X, y in dataloader:
    optimizer.zero_grad()
    loss = criterion(model(X), y)
    loss.backward()
    optimizer.step()  # Internally smooths g into v and updates W
```

---

## 7. Summary

* SGD can oscillate or move slowly in certain directions
* Momentum introduces a velocity term $v$ that **accumulates past gradients**
* Updates become **smoother and faster**, especially in long, narrow valleys
