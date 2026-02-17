"""Utility functions for pion classification neural network."""
import numpy as np
import pickle
from sklearn.metrics import confusion_matrix


def load_file(file_path):
    """Load the data from a pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def count_instances_in_list(list, X: list):
    """Count the number of particles of a given type (or set of types) in a list."""
    count = 0
    for item in list:
        if item in X:
            count += 1
    return count


def classify_X_as_Y(list_a, list_b, X: list, Y: list):
    """Count the number of times a particle of type X is identified as a particle of type Y."""
    X_as_Y = 0
    if type(list_a) != list:
        list_a = list_a.tolist()
    if type(list_b) != list:
        list_b = list_b.tolist()
    for i in range(len(list_a)):
        if list_a[i] in X and list_b[i] in Y:
            X_as_Y += 1
    return X_as_Y


def calculate_uncertainty(k, n):
    """Calculate uncertainty for a metric."""
    metric = k / n
    uncertainty = np.sqrt(metric * (1 - metric) / n)
    return uncertainty


def purity(y_pred, y_test, identified_particles: list, true_particles: list, return_uncertainty=False):
    """Calculate purity - ratio of correctly identified particles to total identified particles."""
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_identified = count_instances_in_list(y_pred, identified_particles)

    if total_identified == 0:
        if return_uncertainty:
            return 0, 0
        else:
            return 0
    else:
        if return_uncertainty:
            return matched / total_identified, calculate_uncertainty(matched, total_identified)
        else:
            return matched / total_identified


def efficiency(y_pred, y_test, identified_particles: list, true_particles: list, return_uncertainty=False):
    """Calculate efficiency - ratio of correctly identified particles to total real particles."""
    matched = classify_X_as_Y(y_pred, y_test, identified_particles, true_particles)
    total_real = count_instances_in_list(y_test, true_particles)
    if total_real == 0:
        if return_uncertainty:
            return 0, 0
        else:
            return 0
    else:
        if return_uncertainty:
            return matched / total_real, calculate_uncertainty(matched, total_real)
        else:
            return matched / total_real


def create_confusion_matrix(y_test, y_pred):
    """Create confusion matrix with purity and efficiency information."""
    labels = sorted(list(set(list(y_test)) | set(list(y_pred))))
    labels = [str(label) for label in labels]
    y_test = [str(label) for label in y_test]
    y_pred = [str(label) for label in y_pred]
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    purities = []
    efficiencies = []
    counts = cm.flatten()

    for true_particle in range(len(labels)):
        for predicted_particle in range(len(labels)):
            pur, pur_uncertainty = purity(y_pred, y_test, [labels[predicted_particle]], [labels[true_particle]], return_uncertainty=True)
            purity_ = f"{100*pur:.1f} ± {100*pur_uncertainty:.1f}%"
            eff, eff_uncertainty = efficiency(y_pred, y_test, [labels[predicted_particle]], [labels[true_particle]], return_uncertainty=True)
            efficiency_ = f"{100*eff:.1f} ± {100*eff_uncertainty:.1f}%"
            purities.append(purity_)
            efficiencies.append(efficiency_)

    info = zip(counts, purities, efficiencies)
    info = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in info]
    info = np.asarray(info).reshape(cm.shape)

    cm = cm[::-1]
    info = info[::-1]

    return cm, info, labels
