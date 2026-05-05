import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import gc
import io

np.random.seed(0)

X = np.linspace(-1, 1, 100)
y = 2 * X + 1 + np.random.randn(100) * 0.2


def loss(w, b):
    y_pred = w * X + b
    return ((y - y_pred) ** 2).mean()


def gradient_step(w, b, lr):
    y_pred = w * X + b
    dw = (-2 * (X * (y - y_pred))).mean()
    db = (-2 * (y - y_pred)).mean()
    w = w - lr * dw
    b = b - lr * db
    return w, b


def train(lr, steps=300):
    w, b = -1, -1
    history = []
    for i in range(steps):
        history.append((w, b))
        w, b = gradient_step(w, b, lr)
    return history


def make_gif(lr):
    history = train(lr)
    frames = []
    total_steps = len(history)

    # Set smaller figure size
    dpi = 150
    figsize = (10, 6)

    for step, (w, b) in enumerate(history):
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.scatter(X, y, s=10, alpha=0.7)
        y_pred = w * X + b
        ax.plot(X, y_pred, "r-", linewidth=1.5)

        current_loss = loss(w, b)
        ax.set_title(
            f"lr={lr}, step={step+1}/{total_steps}\nw={w:.4f}, b={b:.4f}, loss={current_loss:.4f}",
            fontsize=10,
        )

        canvas = FigureCanvas(fig)
        canvas.draw()

        # Save image to buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
        buf.seek(0)

        # Read image from buffer
        img = imageio.imread(buf)
        frames.append(img)

        buf.close()
        plt.close(fig)
        gc.collect()

        # Progress indicator
        if (step + 1) % 50 == 0:
            print(f"lr={lr}: Processed {step+1}/{total_steps} frames")

    # Save GIF
    print(f"lr={lr}: Saving GIF...")
    imageio.mimsave(f"lr_{lr}.gif", frames, duration=0.1, fps=10)
    print(f"lr={lr}: GIF saved with {len(frames)} frames")

    # Clean memory
    del frames
    gc.collect()


if __name__ == "__main__":
    for lr in [0.001, 0.01, 0.03, 1, 2]:
        print(f"\nProcessing lr={lr}")
        make_gif(lr)
        print(f"lr={lr} completed")
