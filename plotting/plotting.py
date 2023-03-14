import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.animation import FuncAnimation
import matplotlib

matplotlib.use('TkAgg')


def animation_plot(lis: list):
    avg_window = 100
    # Initialize the plot
    fig, ax = plt.subplots()
    line, = ax.plot(list(range(len(lis))), lis)
    rolling_avg = pd.Series(lis).rolling(window=avg_window).mean()

    avg, = ax.plot(list(range(len(lis))), rolling_avg, color='red')

    def update():
        line.set_data(list(range(len(lis))), lis)
        rolling_avg = pd.Series(lis).rolling(window=avg_window).mean()
        avg.set_data(list(range(len(lis))), rolling_avg)
        ax.relim()
        ax.autoscale_view(True, True, True)  # update the axes limits
        fig.canvas.draw()  # redraw the plot
        plt.pause(0.001)  # wait for a short time to allow the plot to be shown

    return update