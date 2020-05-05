import matplotlib.pyplot as plt
import numpy as np
import pickle
from matplotlib.colors import LogNorm, Normalize
from matplotlib.widgets import RectangleSelector, LassoSelector
from matplotlib.path import Path
from matplotlib.patches import PathPatch
from enum import Enum, unique
from typing import List, Tuple, Optional
from numpy import ndarray as array
from pathlib import Path as PPath
from .matrix import Matrix
from .turbo import *


@unique
class State(Enum):
    FG = 1
    BG = 2


class Paint:
    """
    Is dependent on the binning. Could be made independent by instead
    storing the path. Moves the problem of to the possibility of a bad path.
    """
    def __init__(self, records: List[Tuple[array, array]] = [],
                 path: Optional[PPath] = None):
        self.records = records
        if path is not None:
            self.load(path)

    def __call__(self, matrix: Matrix) -> None:
        self.act_on(matrix)

    def act_on(self, matrix: Matrix) -> None:
        for bg, fg in self.records:
            self.paint(matrix, bg, fg)

    def paint(self, matrix: Matrix, bg: array, fg: array) -> None:
        random_indices = np.random.choice(len(bg), len(fg))
        bg = bg[random_indices]
        new = matrix.values.T[bg[:, 0], bg[:, 1]]
        matrix.values.T[fg[:, 0], fg[:, 1]] = new

    def save(self, path: PPath) -> None:
        path = PPath(path)
        with path.open("wb") as out:
            pickle.dump(self.records, out)

    def load(self, path: PPath) -> None:
        path = PPath(path)
        with path.open("rb") as inf:
            self.records = pickle.load(inf)


class Painter():
    def __init__(self, matrix, **kwargs):
        self.x = matrix.Ex
        self.y = matrix.Eg
        self.values = matrix.values.T
        self.mat = matrix
        self.plot_kwargs = kwargs

        self.patches = []
        self.peaks = []
        self.peak_numbers = []
        self.state = State.FG
        self.background = []
        self.foreground = []

        self.records: List[Tuple[array, array]] = []

        fig, self.ax = plt.subplots(figsize=(10, 10))
        self.plot_matrix()
        # cid = fig.canvas.mpl_connect("button_press_event", self.click_factory())
        cid = fig.canvas.mpl_connect("key_press_event", self.press_factory())

        def format_coord(x, y):
            xarr = self.x
            yarr = self.y
            if ((x > xarr.min()) & (x <= xarr.max())
               & (y > yarr.min()) & (y <= yarr.max())):
                col = np.searchsorted(xarr, x)-1
                row = np.searchsorted(yarr, y)-1
                z = self.values.T[row, col]
                return f'x={x:1.2f}, y={y:1.2f}, z={z:1.2E}'
                # return f'x={x:1.0f}, y={y:1.0f}, z={z:1.3f}   [{row},{col}]'
            else:
                return f'x={x:1.0f}, y={y:1.0f}'
        self.ax.format_coord = format_coord
        self.ls = LassoSelector(self.ax, self.onselect_factory(), button=[1])

        # For communication

    def plot_matrix(self):
        q = self.ax.pcolormesh(self.x, self.y, self.values.T,
                               cmap="turbo", norm=LogNorm(), **self.plot_kwargs)
        self.quad_mesh = q

    def pick(self):
        values, (x0, x1), (y0, y1) = self.visible()
        (xm, ym), (x, y) = centroid(values)
        # Shift the points back
        xm = int(xm + x0)
        ym = int(ym + y0)
        self.add_peak([xm, ym])
        self.add_rectangles(x, y)
        scat = self.ax.scatter(self.x[xm], self.y[ym], color="r", marker="x")
        self.patches[-1].append(scat)
        self.ax.figure.canvas.draw_idle()

    def press_factory(self):
        def handle_keypress(event):
            # Switch between recording FG and BG
            if event.key == 'b':
                self.swap_state()
            elif event.key == 'x':
                self.paint_fg()
            elif event.key == 'd':
                self.background = []
                self.foreground = []
                for patch in self.patches:
                    patch.remove()
                    del patch
                self.patches = []
                self.ax.figure.canvas.draw_idle()
                self.state = State.FG
            elif event.key == 'r':
                self.redraw()
            elif event.key == 'w':
                matrix = om.Matrix(Eg=self.x, Ex=self.y, values=self.values)
                matrix.save("edited.npy")
            elif event.key == ';':
                plt.set_cmap('viridis')

        return handle_keypress

    def onselect_factory(self):
        def handle_onselect(vertices):
            path = Path(vertices)
            values, (x0, x1), (y0, y1) = self.visible()
            x, y = np.meshgrid(self.x[x0:x1], self.y[y0:y1])
            x, y = x.ravel(), y.ravel()
            xy = np.stack([x, y]).T
            indices = np.nonzero(path.contains_points(xy))[0]
            x = indices // values.shape[0]
            y = indices %  values.shape[0]
            x, y = y, x
            self.add_polygon(path)
            #self.add_rectangles(x, y)
            x += x0
            y += y0
            coords = np.asarray([x, y]).T
            ground = self.foreground if self.isFG() else self.background
            if len(ground) == 0:
                ground = coords
            else:
                ground = np.append(ground, coords, axis=0)

            if self.isFG():
                self.foreground = ground
            else:
                self.background = ground

            self.ax.figure.canvas.draw_idle()
            self.swap_state()

        return handle_onselect

    def add_polygon(self, path):
        c = 'r' if self.isFG() else 'b'
        patch = PathPatch(path, facecolor=c, alpha=0.5, lw=1)
        self.ax.add_patch(patch)
        self.patches.append(patch)

    def visible(self):
        cur_xlim = self.ax.get_xlim()
        cur_ylim = self.ax.get_ylim()
        x0 = index(self.x, cur_xlim[0])
        x1 = index(self.x, cur_xlim[1])
        y0 = index(self.y, cur_ylim[0])
        y1 = index(self.y, cur_ylim[1])

        values = self.values[x0:x1, y0:y1]
        return values, (x0, x1), (y0, y1)

    def isFG(self):
        return self.state == State.FG

    def swap_state(self):
        if self.isFG():
            self.state = State.BG
        else:
            self.state = State.FG
        print("State is ", self.state)

    def paint_fg(self):
        # For elem in self.foreground:
        #     i = np.random.choice(len(self.background))
        #     bg = self.background[i]
        #     self.values[elem[0], elem[1]] = self.values[bg[0], bg[1]]
        I = np.random.choice(len(self.background), len(self.foreground))
        bg = self.background[I]
        new = self.values[bg[:, 0], bg[:, 1]]
        #new = np.random.poisson(new, len(new))

        self.values[self.foreground[:, 0], self.foreground[:, 1]] = new

        self.redraw()

        # Store the used background and foreground
        self.records.append([self.background, self.foreground])

    def redraw(self):
        C = self.values.T[:-1, :-1]
        self.quad_mesh.set_array(C.ravel())
        self.ax.figure.canvas.draw_idle()

    def get_paint(self) -> Paint:
        return Paint(self.records)


def index(arr, e) -> int:
    return np.argmin(abs(arr-e))


if __name__ == "__main__":
    fg = np.fromfile("../zinc/egex_fg.bin", dtype="float32").reshape((-1, 2))
    bg = np.fromfile("../zinc/egex_bg.bin", dtype="float32").reshape((-1, 2))
    nbins = 1000
    bins = np.linspace(0, 12000, nbins)
    hfg, *_ = np.histogram2d(fg[:, 0], fg[:, 1], bins=bins)
    hbg, *_ = np.histogram2d(bg[:, 0], bg[:, 1], bins=bins)
    del fg, bg
    mat = hfg - hbg
    mid = bins + (bins[1]-bins[0])/2
    mid = mid[:-1]
    raw = om.Matrix(values=mat.T, Eg=mid, Ex=mid)
    click = Painter(raw, vmax=1e3)
    plt.show()
