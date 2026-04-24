# Breaking Changes in Neural Network History

## Why Deep Learning Suddenly Started Working

Deep learning appears modern, but many of its core ideas were discovered decades ago. The real story of neural networks is not a sequence of completely new ideas, but a series of **breaking changes** that made previously impractical ideas suddenly usable.

A useful way to understand the history of neural networks is to separate two questions:

1. **What neural networks can theoretically do**
2. **What we can actually train in practice**

For decades, theory suggested neural networks were extremely powerful. But practical training remained very difficult.

---

# 1. Theoretical Power Was Known Early

In 1989, the **Universal Approximation Theorem** showed that neural networks could approximate arbitrary functions.

For a neural network with a single hidden layer:

$$f(x) = \sum_{i=1}^{N} v_i \sigma(w_i^T x + b_i)$$

Cybenko (1989) and Hornik (1989) proved that such networks can approximate any continuous function on a compact domain.

In other words:

> Neural networks were already known to be powerful in the 1980s.

However, this theorem did **not** tell us how to efficiently train deep networks. The main obstacles were:

* Vanishing gradients
* Slow computation
* Poor initialization
* Lack of large datasets

Deep learning therefore remained largely dormant for decades.

---

# Breaking Change #1: ReLU (Rediscovered)

The **Rectified Linear Unit (ReLU)** activation function is defined as:

$$\text{ReLU}(x) = \max(0, x)$$

ReLU itself is **not new**. Similar functions appeared much earlier:

| Year | Event |
| --- | --- |
| 1969 | Piecewise linear activations discussed in early neural models |
| 1980s | Variants used in some early neural research |
| 2010 | Nair & Hinton show ReLU works well in deep networks |
| 2011 | Glorot & Bengio analyze ReLU theoretically |

Thus, ReLU was **rediscovered and popularized**, not invented.

## Why ReLU Was a Breaking Change

Before ReLU, neural networks typically used the **sigmoid** activation function:

$$\sigma(x) = \frac{1}{1 + e^{-x}}$$

Its derivative is:

$$\sigma'(x) = \sigma(x)(1-\sigma(x))$$

The derivative is **bounded by 0.25**. When training deep networks, gradients multiply through many layers via the chain rule:

$$\frac{\partial L}{\partial x_1} = \frac{\partial L}{\partial x_n} \prod_{i=1}^{n} \frac{\partial x_i}{\partial x_{i-1}}$$

If each layer roughly contributes a factor around **0.25**, then after 10 layers:

$$0.25^{10} \approx 9.5 \times 10^{-7}$$

The gradient becomes extremely small. This is known as the **vanishing gradient problem**.

ReLU changes this dramatically. The derivative of ReLU is:

$$\frac{d}{dx}\text{ReLU}(x)= \begin{cases} 1 & x>0 \\ 0 & x\le 0 \end{cases}$$

For positive activations, gradients are perfectly preserved: $1 \times 1 \times 1 \times \dots$

Thus, gradients can flow through many layers. This simple function made **deep networks trainable**.

---

# Breaking Change #2: Convolutional Neural Networks (CNNs)

Convolutional neural networks were introduced by **Yann LeCun** in the 1990s.

The convolution operation is:

$$y(i,j)=\sum_{m,n} x(i+m,j+n) w(m,n)$$

CNNs introduced three important ideas:

1. **Local receptive fields**
2. **Weight sharing**
3. **Translation invariance**

This dramatically reduced the number of parameters.

**Example comparison:**

| Model | Parameters |
| --- | --- |
| Fully connected (image) | millions |
| CNN | thousands |

CNNs enabled early successes such as LeNet (1998) for digit recognition. However, CNNs alone were **not enough** to trigger the deep learning revolution. Training deeper networks was still unstable.

---

# Breaking Change #3: GPU Computing

Training neural networks requires massive matrix multiplication:

$$Y = WX$$

GPUs are designed for **parallel numerical computation**, which makes them ideal for deep learning. The key moment was the **2012 ImageNet breakthrough**. The model **AlexNet** used GPUs, ReLU, dropout, and large datasets.

Training that would take **months on CPUs** could be done in **days on GPUs**. This drastically shortened the research iteration cycle:

* **CPU era:** Idea → Experiment → Results (months)
* **GPU era:** Idea → Experiment → Results (days)

This accelerated innovation dramatically.

---

# Breaking Change #4: Optimization Algorithms

Early neural networks were trained with **plain stochastic gradient descent (SGD)**:

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

But SGD is highly sensitive to the learning rate, surface curvature, and noisy gradients. Several improvements were developed to fix this.

## Momentum

Momentum smooths updates using velocity:

$$v_t = \beta v_{t-1} + (1-\beta)\nabla L(\theta_t)$$

$$\theta_{t+1} = \theta_t - \eta v_t$$

This accelerates training in consistent directions.

## Adam Optimizer

Adam (2014) combines **momentum** and **adaptive learning rates**.

First moment estimate:


$$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$$

Second moment estimate:


$$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$$

Update rule:


$$\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon}$$

Adam quickly became the **default optimizer** for many deep learning models.

---

# Breaking Change #5: Batch Normalization

Training deep networks suffers from **internal covariate shift**. Batch normalization stabilizes layer inputs.

For each mini-batch, calculate the mean and variance:

$$\mu_B = \frac{1}{m}\sum_{i=1}^{m} x_i$$

$$\sigma_B^2 = \frac{1}{m}\sum_{i=1}^{m}(x_i-\mu_B)^2$$

Normalize the inputs:

$$\hat{x_i}=\frac{x_i-\mu_B}{\sqrt{\sigma_B^2+\epsilon}}$$

Then apply learned scale and shift parameters:

$$y_i = \gamma \hat{x_i} + \beta$$

BatchNorm allows for higher learning rates, faster convergence, and much deeper networks.

---

# Breaking Change #6: Initialization (He Initialization)

Poor initialization can cause gradients to vanish or explode before training even begins. He initialization is specifically designed for ReLU networks:

$$W \sim \mathcal{N}\left(0,\frac{2}{n}\right)$$

where $n$ is the number of input neurons. This keeps activation variance stable across layers. Without proper initialization, even ReLU networks fail to train.

---

# Breaking Change #7: Residual Networks (ResNet)

Even with ReLU and BatchNorm, very deep networks still struggled to converge. ResNet introduced **residual connections**.

Instead of learning a direct mapping $H(x)$, the network learns a residual mapping:

$$F(x) = H(x) - x$$

So the output becomes:

$$y = F(x) + x$$

This allows gradients to flow directly back through **identity shortcuts**. Residual connections enabled networks with 50, 101, and even 152 layers to train successfully.

---

# The Real Lesson

Deep learning did not succeed because of **one single invention**. Instead, it required several **breaking changes working together**.

| Breakthrough | Year | Impact |
| --- | --- | --- |
| Universal Approximation | 1989 | Proved neural networks could approximate any function |
| CNN (LeNet) | 1998 | Parameter sharing made vision tasks feasible |
| GPU Deep Learning | ~2010–2012 | Enabled large-scale parallel training |
| ReLU (rediscovered) | 2010–2011 | Solved vanishing gradients in deep networks |
| Better Optimization | 1990s–2014 | Momentum, RMSProp, and Adam stabilized training |
| Batch Normalization | 2015 | Allowed much deeper networks to converge |
| He Initialization | 2015 | Maintained activation variance from the start |
| ResNet | 2015 | Enabled extremely deep architectures via shortcuts |

One important detail: **ReLU is historically unusual.**

The function itself is extremely simple and had appeared in earlier neural models. However, it was **largely ignored for decades**. Around **2010–2011**, several papers demonstrated that ReLU dramatically improved deep network training. So ReLU was not really *invented* in 2010—it was **rediscovered and finally understood**.

---

# A Deeper Interpretation

If we step back and look at all these breakthroughs, something interesting appears. Almost every major change in deep learning history is actually about **gradients**.

Training a neural network ultimately means solving the same problem: *How do we find good parameters $W$ and $b$?* In other words, we want to minimize a loss function $L(W,b)$ using gradient-based optimization.

Every major breakthrough improved **how gradients behave during training**.

| Technique | What it fixes |
| --- | --- |
| **ReLU** | Prevents vanishing gradients |
| **Initialization** | Keeps gradients stable at the start |
| **BatchNorm** | Stabilizes gradient distribution |
| **Adam / Momentum** | Improves gradient updates |
| **ResNet** | Creates shortcut paths for gradients |

Even GPUs are indirectly part of this story: they allow us to compute gradients **millions of times faster**.

---

# The Simplest View of Deep Learning

At the end of the day, deep learning is surprisingly simple. A neural network is just a very large function:

$$y = f(x; W, b)$$

Training means adjusting $W$ and $b$ so that predictions match the data. Everything else—architectures, mathematical tricks, optimizers—exists solely to make **this search process possible**.
