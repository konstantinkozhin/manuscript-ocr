import numpy as np
import matplotlib.pyplot as plt

def show_histogram(scores, title="Score Histogram"):
    plt.figure(figsize=(6, 4))
    plt.hist(scores, bins=30, color='blue', alpha=0.7)
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title(title)
    plt.grid(True)
    plt.show()
