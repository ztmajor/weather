import numpy as np
import matplotlib.pyplot as plt


def draw_h(data, predictions, length):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = 15
    fig_size[1] = 5
    plt.rcParams["figure.figsize"] = fig_size

    plt.title('Month vs Passenger')
    plt.ylabel('Total Passengers')
    plt.xlabel('Months')
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(data)
    # plt.show()

    x = np.arange(0, length, 1)
    # print("x", x)
    plt.grid(True)
    plt.autoscale(axis='x', tight=True)
    plt.plot(x, predictions)
    plt.show()
