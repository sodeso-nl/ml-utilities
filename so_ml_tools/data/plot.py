import matplotlib.pyplot as _plt


def xy_data_with_label(x, y) -> None:
    """
    Plots a graph of the values of x whereas x contains vectors of x/y coordinates and y
    is the label (0 or 1).

    Args:
        x: is an array containing vectors of x/y coordinates.
        y: are the associated labels (0=blue, 1=red)
    """
    _plt.plot(x[:, 0][y == 1], x[:, 1][y == 1], "bs")
    _plt.plot(x[:, 0][y == 0], x[:, 1][y == 0], "r^")

    # X contains two features, x1 and x2
    _plt.xlabel(r"$x_1$", fontsize=20)
    _plt.ylabel(r"$x_2$", fontsize=20)

    # Displaying the plot.
    _plt.show()
