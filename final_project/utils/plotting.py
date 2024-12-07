from typing import List

import matplotlib.pyplot as plt
import numpy as np

def moving_average(data: np.array, window_size: int = 50) -> np.array:
    """Smooths 1-D data array using a moving average.

    Args:
        data: data to plot
        window_size: Size of the smoothing window

    Returns:
        smooth_data: A 1-d numpy.array with the same size as data
    """
    kernel = np.ones(window_size)
    smooth_data = np.convolve(data, kernel) / np.convolve(
        np.ones_like(data), kernel
    )
    return smooth_data[: -window_size + 1]

def plot_curves(arr_list: List[np.array], legend_list: List[str], color_list: List[str], ylabel: str, title: str = None, xlabel: str = "Episodes"):
    """
    Args:
        arr_list: list of results arrays to plot
        legend_list: list of legends corresponding to each result array
        color_list: list of color corresponding to each result array
        ylabel: label for the y axis
        title: plot title
        xlabel: label for the x axis
    """
    # set the figure type
    plt.clf()
    fig, ax = plt.subplots()

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)

    # ploth results
    h_list = []
    for arr, legend, color in zip(arr_list, legend_list, color_list):
        # compute the standard error
        arr_err = arr.std(axis=0) / np.sqrt(arr.shape[0])
        # plot the mean
        h, = ax.plot(range(0, 1000*arr.shape[1], 1000), arr.mean(axis=0), color=color, label=legend)
        # plot the confidence band
        arr_err = 1.96 * arr_err
        ax.fill_between(range(arr.shape[1]), arr.mean(axis=0) - arr_err, arr.mean(axis=0) + arr_err, alpha=0.3,
                        color=color)
        # save the plot handle
        h_list.append(h)

    # plot legends
    ax.legend(handles=h_list)
    ax.set_title(title)
    plt.show()