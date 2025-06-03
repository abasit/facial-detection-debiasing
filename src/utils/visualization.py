import matplotlib.pyplot as plt

def plot_demographic_results(results, title="Model Predictions by Demographic Group"):
    """
    Plot demographic evaluation results.

    Args:
        results (dict): Demographic names -> average probabilities
        title (str): Plot title
    """
    names = list(results.keys())
    probs = list(results.values())

    plt.figure(figsize=(10, 6))
    xx = range(len(names))
    plt.bar(xx, probs)
    plt.xticks(xx, names, rotation=45)
    plt.ylabel('Average Prediction Probability')
    plt.title(title)
    plt.ylim(max(0, min(probs) - 0.1), min(1.0, max(probs) + 0.1))
    plt.tight_layout()
    plt.show()
