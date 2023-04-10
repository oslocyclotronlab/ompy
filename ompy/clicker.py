from __future__ import annotations
from matplotlib.widgets import TextBox
from scipy.optimize import minimize
from iminuit import Minuit
from matplotlib.widgets import EllipseSelector, RectangleSelector, _SelectorWidget
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.pyplot as plt
import sys
from . import Matrix
from .detector import Detector, Oslo, CompoundDetector
from .stubs import Axes
import numpy as np
from pathlib import Path

# define normalized 2D gaussian
# @njit


def gaus2d(x, y, mx=0, my=0, sx=1, sy=1, A=1):
    return A / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.)
                                                 + (y - my)**2. / (2. * sy**2.)))


class Clicker:
    def __init__(self, matrix: Matrix, detector: Detector = Oslo()):
        self.mat = matrix
        self.selectors: list[_SelectorWidget] = []
        self.active_selector: _SelectorWidget = None
        self.els: EllipseSelector | None = None
        self.rts: RectangleSelector | None = None
        self.ax = None
        self.detector: Detector = detector
        self.areas: list[Area] = []
        self.savepath = Path('areas')

    def start(self):
        fig, ax = plt.subplots()
        self.ax = ax

        self.mat.plot(ax=ax)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        # cid = fig.canvas.mpl_connect('button_press_event', onclick)

        self.rts = RectangleSelector(ax, self.select_callback,
                                     useblit=True, minspanx=5, minspany=5,
                                     spancoords='pixels', interactive=True)
        self.selectors.append(self.rts)
        self.active_selector = self.rts
        self.els = EllipseSelector(ax, self.select_callback,
                                   useblit=True, minspanx=5, minspany=5,
                                   spancoords='pixels', interactive=True)
        self.els.set_active(False)
        self.selectors.append(self.els)
        fig.canvas.mpl_connect('key_press_event', self.toggle_selector)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    def select_callback(self, eclick, erelease):
        if eclick.button == 1:
            x1, y1 = eclick.xdata, eclick.ydata
            x2, y2 = erelease.xdata, erelease.ydata
            print(f"({x1:3.2f}, {y1:3.2f}) --> ({x2:3.2f}, {y2:3.2f})")
            print(
                f"The buttons you used were: {eclick.button} {erelease.button}")

    def toggle_selector(self, event):
        match event.key:
            case 't':
                for selector in self.selectors:
                    name = type(selector).__name__
                    if selector.active:
                        print(f'{name} deactivated.')
                        selector.set_active(False)
                    else:
                        print(f'{name} activated.')
                        selector.set_active(True)
                        self.active_selector = selector
            case 'y':
                if not self.active_selector.visible:
                    print('No selector active.')
                    return
                if self.active_selector == self.els:
                    raise NotImplementedError()
                else:
                    mat = self.filter_rectangle(event)
                    area = Area(mat)
                    res = area.fit_minuit()
                    fig, ax = plt.subplots()
                    mat.plot(ax=ax)
                    X, Y = mat.Y, mat.X
                    Z = mat.values
                    XX, YY = np.meshgrid(X, Y)
                    print(res)
                    mx, my, sx, sy, A = res.values
                    # mx, my, sx, sy, A = res.x
                    zhat = gaus2d(XX, YY, mx, my, sx, sy, 1)
                    ax.plot(mx, my, 'rx')
                    levels = np.array([0.68, 0.95, 0.997])
                    ax.contour(XX, YY, zhat)  # , levels=levels)
                    ax.contour(XX, YY, zhat, levels=levels, cmap='turbo')

                    # plot as 3d plot
                    fig = plt.figure()
                    ax = fig.add_subplot(
                        211, projection='3d')
                    ax2 = fig.add_subplot(
                        212, projection='3d', computed_zorder=False)

                    def on_move(event):
                        if event.inaxes == ax:
                            ax2.view_init(elev=ax.elev, azim=ax.azim)
                        elif event.inaxes == ax2:
                            ax.view_init(elev=ax2.elev, azim=ax2.azim)
                        else:
                            return
                        fig.canvas.draw_idle()

                    c1 = fig.canvas.mpl_connect('motion_notify_event', on_move)

                    ax.plot_surface(XX, YY, Z, cmap='turbo')
                    X_ = np.linspace(X[0], X[-1], 100)
                    Y_ = np.linspace(Y[0], Y[-1], 100)
                    XX_, YY_ = np.meshgrid(X_, Y_)
                    zhat = gaus2d(XX_, YY_, mx, my, sx, sy, A)
                    print(zhat.shape)
                    # ax.contour(XX_, YY_, zhat, cmap='turbo_r', zorder=5.5)
                    ax2.plot_surface(XX_, YY_, zhat, cmap='turbo', zorder=4.4)
                    # ax.plot_wireframe(XX_, YY_, zhat, cmap='turbo_r',
                    #                  zorder=5.5, rstride=10, cstride=10)
                sys.stdout.flush()
            case 'u':
                if not self.active_selector.visible:
                    print('No selector active.')
                    return
                mat = self.filter_rectangle(event)
                area = Area(mat)
                area.plot_box(self.ax, facecolor='blue')
                self.ax.figure.canvas.draw_idle()
                self.areas.append(area)
                self.savepath.mkdir(parents=True, exist_ok=True)
                area.save(self.savepath / f'area_{len(self.areas)}.npz')

    def filter_rectangle(self, event) -> Matrix:
        xmin, xmax, ymin, ymax = self.rts.extents
        mat = self.mat.vloc[ymin:ymax, xmin:xmax]
        return mat

    def filter_ellipse(self, event):
        es = self.els
        cx, cy = es.center  # tuple of floats: (x, y)

        # calculating the width and height
        # self.es.extents returns tuple of floats: (xmin, xmax, ymin, ymax)
        xmin, xmax, ymin, ymax = self.els.extents
        width = xmax - xmin
        height = ymax - ymin
        print(f'center=({cx:.2f},{cy:.2f}), '
              f'width={width:.2f}, height={height:.2f}')
        # 2. Create an ellipse patch
        # CAUTION: DO NOT PLOT (==add this patch to ax), as the coordinates will
        # be transformed and you will not be able to directly check your data
        # points.
        ellipse = Ellipse((cx, cy), width, height)

    def load_areas(self, path) -> None:
        path = Path(path)
        for f in path.glob('*.npz'):
            self.areas.append(Area.from_path(f))


class Area:
    def __init__(self, area: Matrix):
        self.area = area

    def initial_heuristic(self, detector: CompoundDetector = Oslo()):
        mx = self.X.mean()
        my = self.Y.mean()
        sx = detector.egdetector._sigma(mx)
        sy = detector.exdetector._sigma(my)
        A = self.area.sum()
        p0 = [mx, my, sx, sy, A]
        bounds = [(self.X.min(), self.X.max()),
                  (self.Y.min(), self.Y.max()),
                  (0.1*sx, 1e3*sx),
                  (0.1*sy, 1e3*sy),
                  (1, 1e4*A)]
        return p0, bounds

    def fit_minuit(self, **kwargs):
        Z = self.area.values
        X, Y = np.meshgrid(self.X, self.Y)

        def loss(p):
            mux, muy, sx, sy, A = p
            zhat = gaus2d(X, Y, mux, muy, sx, sy, A)
            diff = Z - zhat
            cost = np.sum(diff**2)
            # print(p, cost)
            # neglog = (Z - zhat)**2 / Z
            # d = 0
            # if sx < 1 or sy < 1:
            #    return 1e6
            return cost  # np.sum(neglog)
        p0, bounds = self.initial_heuristic()
        # res = minimize(loss, p0, bounds=bounds)
        m = Minuit(loss, p0)
        m.limits = bounds
        res = m.migrad(iterate=100)
        return res

    def plot_box(self, ax: Axes, **kwargs) -> Axes:
        """ Plot a rectangle patch """
        xmin, xmax, ymin, ymax = self.extents
        kwargs = {'fill': False, 'edgecolor': 'red'} | kwargs
        print(kwargs)
        rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, **kwargs)
        print(rect)
        ax.add_patch(rect)
        return ax

    @property
    def extents(self) -> tuple[float, float, float, float]:
        """ Return the extents of the area """
        xmin, xmax = self.X[0], self.X[-1]
        ymin, ymax = self.Y[0], self.Y[-1]
        return xmin, xmax, ymin, ymax

    @property
    def X(self) -> np.ndarray:
        return self.area.Y

    @property
    def Y(self) -> np.ndarray:
        return self.area.X

    def save(self, path) -> None:
        path = Path(path)
        self.area.save(path)

    @staticmethod
    def from_path(path) -> Area:
        path = Path(path)
        mat = Matrix.from_path(path)
        return Area(mat)


class Fit:
    def __init__(self, area: Matrix, mx, my, sx, sy, A):
        self.area = area
        self.mx = mx
        self.my = my
        self.sx = sx
        self.sy = sy
        self.A = A

    def eval(self, n: None | int = None) -> Matrix:
        X, Y = self.area.X, self.area.Y
        if n is not None:
            X = np.linspace(X.min(), X.max(), n)
            Y = np.linspace(Y.min(), Y.max(), n)
        XX, YY = np.meshgrid(X, Y)
        zhat = gaus2d(XX, YY, self.mx, self.my, self.sx, self.sy, self.A)
        return Matrix(X=X, Y=Y, values=zhat)

    def plot_area(self, ax: Axes | None = None, **kwargs):
        return self.area.plot(ax=ax, **kwargs)

    def plot_fit_3d(self, ax: Axes | None = None, **kwargs):
        if ax is None:
            fig = plt.subplots()
            ax = fig.add_subplot(111, projection='3d')
        zhat = self.eval()
        ax.plot_surface(zhat.X, zhat.Y, zhat.values, **kwargs)
        return ax

    def plot_fit(self, ax: Axes | None = None, **kwargs):
        ax = self.area.plot(ax=ax)
        zhat = self.eval()
        ax.contour(zhat.X, zhat.Y, zhat.values, **kwargs)
        return ax
