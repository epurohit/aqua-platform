"""
Dataset generators and loaders for training and evaluation.
"""
import numpy as np
from typing import Tuple


def make_spirals(n_samples: int = 1200, n_classes: int = 3, noise: float = 0.2, seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates synthetic 2D multi-class spirals dataset.

    Args:
        n_samples (int): Total number of samples.
        n_classes (int): Number of spiral arms / classes.
        noise (float): Gaussian noise magnitude.
        seed (int): Random seed for reproducibility.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Features X of shape (N, 2) and labels y of shape (N,).
    """
    np.random.seed(seed)
    samples_per_class = n_samples // n_classes
    X = np.zeros((samples_per_class * n_classes, 2))
    y = np.zeros(samples_per_class * n_classes, dtype=int)
    
    for class_number in range(n_classes):
        ix = range(samples_per_class * class_number, samples_per_class * (class_number + 1))
        r = np.linspace(0.0, 1.0, samples_per_class)
        t = np.linspace(class_number * 4, (class_number + 1) * 4, samples_per_class) + np.random.randn(samples_per_class) * noise
        X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
        y[ix] = class_number
        
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return X[indices], y[indices]


def load_digits_dataset(seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Loads standard Scikit-Learn Digits classification dataset (8x8 images, 10 classes).

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: (X_train, y_train, X_test, y_test)
    """
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    
    digits = load_digits()
    X = digits.data.astype(np.float64) / 16.0  # Normalize to [0, 1]
    y = digits.target.astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    return X_train, y_train, X_test, y_test
