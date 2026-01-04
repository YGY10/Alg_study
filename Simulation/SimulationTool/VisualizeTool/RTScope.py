import matplotlib.pyplot as plt


class RealtimeScope2D:
    def __init__(self, title, xlabel="x", ylabel="y", max_points=1000):
        plt.ion()

        self.fig, self.ax = plt.subplots()
        self.ax.set_title(title)
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)
        self.ax.grid(True)

        self.max_points = max_points
        self.lines = {}  # name -> line
        self.data = {}  # name -> (x_list, y_list)

    def add_channel(self, name, color=None, linestyle="-"):
        (line,) = self.ax.plot([], [], linestyle=linestyle, label=name, color=color)
        self.lines[name] = line
        self.data[name] = ([], [])
        self.ax.legend()

    def update(self, name, x, y):
        if name not in self.data:
            raise ValueError(f"Channel '{name}' not registered")

        xdata, ydata = self.data[name]
        xdata.append(x)
        ydata.append(y)

        if len(xdata) > self.max_points:
            self.data[name] = (xdata[-self.max_points :], ydata[-self.max_points :])

        self.lines[name].set_data(self.data[name][0], self.data[name][1])
        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.001)

    def hold(self):
        plt.ioff()
        plt.show()


class RealtimeScope3D:
    def __init__(self, title, labels=("x", "y", "z"), max_points=1000):
        plt.ion()

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection="3d")
        (self.line,) = self.ax.plot([], [], [], lw=2)

        self.ax.set_title(title)
        self.ax.set_xlabel(labels[0])
        self.ax.set_ylabel(labels[1])
        self.ax.set_zlabel(labels[2])

        self.xdata = []
        self.ydata = []
        self.zdata = []
        self.max_points = max_points

    def update(self, x, y, z):
        self.xdata.append(x)
        self.ydata.append(y)
        self.zdata.append(z)

        if len(self.xdata) > self.max_points:
            self.xdata = self.xdata[-self.max_points :]
            self.ydata = self.ydata[-self.max_points :]
            self.zdata = self.zdata[-self.max_points :]

        self.line.set_data(self.xdata, self.ydata)
        self.line.set_3d_properties(self.zdata)

        self.ax.relim()
        self.ax.autoscale_view()

        plt.pause(0.001)
