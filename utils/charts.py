import numpy as np
import matplotlib.pyplot as plt


def radar(scores):
    """
    Creates a radar chart from a dictionary of scores.

    Example:
    scores = {
        "Semantic": 85,
        "Citation": 72,
        "Entity": 90,
        "Contradiction": 100
    }
    """

    labels = list(scores.keys())
    values = list(scores.values())

    # Close the polygon
    labels.append(labels[0])
    values.append(values[0])

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    fig, ax = plt.subplots(
        figsize=(6, 6),
        subplot_kw={"polar": True}
    )

    # Plot outline
    ax.plot(
        angles,
        values,
        linewidth=2
    )

    # Fill area
    ax.fill(
        angles,
        values,
        alpha=0.25
    )

    # Axis labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels[:-1], fontsize=10)

    # Score range
    ax.set_ylim(0, 100)

    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(
        ["20", "40", "60", "80", "100"],
        fontsize=8
    )

    ax.set_title(
        "Reliability Radar",
        fontsize=14,
        pad=20
    )

    ax.grid(True)

    return fig
