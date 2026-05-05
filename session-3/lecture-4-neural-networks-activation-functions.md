# Neural Networks — Activation Functions

![](./img/activation-functions.jpg)

---

## 1. What Happens Without Activation Functions?

If we only apply linear transformations:

$$
a^{(l)} = a^{(l-1)} W^{(l)} + b^{(l)}
$$

Stacking multiple layers without activation:

$$
a^{(L)} = (\dots((x W^{(1)} + b^{(1)}) W^{(2)} + b^{(2)}) \dots) W^{(L)} + b^{(L)}
$$

Collapses to a single linear transformation:

$$
a^{(L)} = x W_{\text{total}} + b_{\text{total}}
$$

**Conclusion:** Without nonlinearity, the network is equivalent to a single-layer linear model, no matter how deep.

---

## 2. Common Activation Functions

### Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

* Output range: (0, 1)
* Squeezes values → good for probabilities


### Tanh

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

* Output range: (-1, 1)
* Centered at zero → helps with optimization


### ReLU (Rectified Linear Unit)

$$
\text{ReLU}(z) = \max(0, z)
$$

* Output range: [0, ∞)
* Sparse activation → many neurons inactive
* Simple and efficient

---

## 3. Why ReLU is Popular

* Simple to compute
* Effective in practice for deep networks
* Encourages sparse representations

---

## 4. Key Takeaway

**Activation functions introduce nonlinearity**:

$$
a^{(l)} = g^{(l)}(a^{(l-1)} W^{(l)} + b^{(l)})
$$

Without $g$, the network cannot learn complex functions. Nonlinearity is what allows neural networks to approximate arbitrary mappings from input to output.

