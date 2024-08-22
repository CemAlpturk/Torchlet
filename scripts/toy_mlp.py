from typing import Generator
import os
import argparse
import numpy as np

from torchlet.tensor import Tensor
from torchlet.nn import Linear, ReLU, Sequential

try:
    import matplotlib.pyplot as plt
    from matplotlib import colormaps as cm
    from tqdm import tqdm
except ImportError as e:
    if "matplotlib" in str(e):
        raise ImportError(
            "matplotlib is required for this script. Install it using 'poetry install --with extras'"
        )
    if "tqdm" in str(e):
        raise ImportError(
            "tqdm is required for this script. Install it using 'poetry install --with extras'"
        )


SAVE_PATH = os.path.join(
    os.path.dirname(__file__),
    "toy_mlp_dataset.png",
)


# Generate toy dataset
def make_moons(
    n_samples: int = 100,
    *,
    shuffle: bool = True,
    noise: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Make two interleaving half circles.
    """
    n_samples_in = n_samples // 2
    n_samples_out = n_samples - n_samples_in

    outer_circ_x = np.cos(np.linspace(0, np.pi, n_samples_out))
    outer_circ_y = np.sin(np.linspace(0, np.pi, n_samples_out))
    inner_circ_x = 1 - np.cos(np.linspace(0, np.pi, n_samples_in))
    inner_circ_y = 1 - np.sin(np.linspace(0, np.pi, n_samples_in)) - 0.5

    X = np.vstack(
        [
            np.append(outer_circ_x, inner_circ_x),
            np.append(outer_circ_y, inner_circ_y),
        ]
    ).T
    y = np.hstack(
        [
            np.zeros(n_samples_out, dtype=np.intp),
            np.ones(n_samples_in, dtype=np.intp),
        ]
    )

    if shuffle:
        idx = np.random.permutation(n_samples)
        X, y = X[idx], y[idx]

    if noise is not None:
        X += np.random.normal(scale=noise, size=X.shape)

    return X, y


def train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    *,
    test_size: float = 0.3,
    verbose: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the dataset into training and testing sets.
    """

    assert test_size > 0 and test_size < 1, "test_size must be between 0 and 1"

    n_samples = X.shape[0]
    n_test_samples = int(n_samples * test_size)

    X_train, X_test = np.split(X, [-n_test_samples])
    y_train, y_test = np.split(y, [-n_test_samples])

    if verbose:
        print(f"X_train.shape: {X_train.shape}")
        print(f"y_train.shape: {y_train.shape}")
        print(f"X_test.shape: {X_test.shape}")
        print(f"y_test.shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test


def get_batch_fn(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
) -> Generator[tuple[Tensor, Tensor], None, None]:
    """
    Infinitaly generate batches of data.
    """
    n_samples = X.shape[0]

    if batch_size > n_samples:
        raise ValueError("Batch size must be smaller than the number of samples")

    while True:
        idx = np.random.permutation(n_samples)
        X, y = X[idx], y[idx]

        for i in range(0, n_samples, batch_size):
            yield Tensor(X[i : i + batch_size]), Tensor(y[i : i + batch_size])


def main(args: argparse.Namespace) -> None:

    np.random.seed(args.seed)

    # Genearate the dataset
    X, y = make_moons(n_samples=args.n_samples, noise=args.noise)
    y = y * 2 - 1  # make y be -1 or 1
    if not args.silent:
        print(f"X.shape: {X.shape}")
        print(f"y.shape: {y.shape}")

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X=X,
        y=y,
        test_size=args.test_size,
        verbose=not args.silent,
    )
    # X_train = X
    # y_train = y
    # X_test = X
    # y_test = y

    # Define the model based on num hidden layers and hidden size
    n_in = X.shape[1]
    n_out = 1  # Binary classification
    hidden_layers = args.n_layers
    hidden_size = args.hidden_size

    layers = [Linear(n_in, hidden_size), ReLU()]

    for _ in range(hidden_layers - 1):
        layers.append(Linear(hidden_size, hidden_size))
        layers.append(ReLU())

    layers.append(Linear(hidden_size, n_out))

    model = Sequential(*layers)

    if not args.silent:
        print(model)
        print("number of parameters", sum(p.size() for p in model.parameters()))

    # Training loop
    alpha = args.alpha
    batch_size = args.batch_size or len(X_train)
    batch_fn = get_batch_fn(X_train, y_train, batch_size)
    for step in (pbar := tqdm(range(args.n_steps), disable=args.silent)):
        # Sample a batch of data
        X_batch, y_batch = next(batch_fn)

        # Compute the forward pass
        y_pred = model(X_batch)[:, 0]
        data_loss = (1 - y_batch * y_pred).relu().mean()

        # L2 regularization
        reg_loss = alpha * sum((p**2).sum() for p in model.parameters())
        loss = data_loss + reg_loss

        # Compute the backward pass
        model.zero_grad()
        loss.backward()

        # Update the parameters
        lr = 1.0 - 0.9 * step / args.n_steps
        for param in model.parameters():
            param.data -= lr * param.grad

        # Update the progress bar
        if step % 100 == 0:
            acc = ((y_pred > 0) == (y_batch > 0)).mean().item()
            pbar.set_description(f"Loss: {loss.item():.4f}, Acc: {acc:.2f}")

    # Test the model
    y_pred = model(Tensor(X_test))[:, 0]
    acc = ((y_pred > 0) == (y_test > 0)).mean().item()
    print(f"Test Accuracy: {acc:.2f}")

    # Plot the dataset and results
    if not args.no_plot:

        # visualize decision boundary

        h = 0.25
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Xmesh = np.c_[xx.ravel(), yy.ravel()]

        inputs = Tensor(Xmesh)
        scores = model(inputs)[:, 0].to_numpy()
        Z = scores > 0
        Z = Z.reshape(xx.shape)

        cmap = cm.get_cmap("Spectral")

        plt.figure()
        plt.contourf(xx, yy, Z, cmap=cmap, alpha=0.8)
        plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=cmap)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())

        plt.savefig(SAVE_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    data_parser = parser.add_argument_group("Dataset")
    data_parser.add_argument(
        "--n_samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    data_parser.add_argument(
        "--noise",
        type=float,
        default=0.1,
        help="Standard deviation of Gaussian noise to add",
    )
    data_parser.add_argument(
        "--test_size",
        type=float,
        default=0.3,
        help="Fraction of samples to include in the test split",
    )

    model_parser = parser.add_argument_group("Model")
    model_parser.add_argument(
        "--hidden_size",
        type=int,
        default=16,
        help="Number of neurons in the hidden layer.",
    )
    model_parser.add_argument(
        "--n_layers",
        type=int,
        default=2,
        help="Number of hidden layers.",
    )

    training_parser = parser.add_argument_group("Training")
    training_parser.add_argument(
        "--n_steps",
        type=int,
        default=1000,
        help="Number of training steps.",
    )
    training_parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Number of samples in each batch.",
    )
    training_parser.add_argument(
        "--alpha",
        type=float,
        default=1e-4,
        help="L2 regularization strength.",
    )

    parser.add_argument(
        "--silent",
        action="store_true",
        help="Suppress all output.",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Do not plot the dataset or results.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    args = parser.parse_args()

    main(args)
