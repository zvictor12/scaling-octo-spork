"""Анимация полей и график непрерывных собственных значений."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np


class Plotter:
    def __init__(self, directory: Path):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)

    def profile_animation(self, data: dict, interval: int = 80) -> tuple:
        """Показывает карты N и rho; кадры не сохраняются на диск."""
        fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
        names = list(data)
        images = [[None, None], [None, None]]
        for row in range(2):
            fields = []
            for model, result in data.values():
                part = result.y[:model.size] if row == 0 else result.y[model.size:]
                fields.append(part)
            low = min(float(np.min(values)) for values in fields)
            high = max(float(np.max(values)) for values in fields)
            for column, (name, (model, result)) in enumerate(data.items()):
                m = model.p.cells
                values = fields[column][:, 0].reshape(m, m).T
                image = axes[row, column].imshow(values, origin="lower",
                    extent=(0, model.p.length, 0, model.p.length),
                    vmin=low, vmax=high, cmap="viridis", interpolation="nearest")
                images[row][column] = image
                axes[row, column].set_xlabel("x")
                axes[row, column].set_ylabel("y")
                if row == 0:
                    axes[row, column].set_title(f"{name}: N, t={result.t[0]:.1f}")
                else:
                    axes[row, column].set_title(r"$\rho$")
            fig.colorbar(images[row][0], ax=axes[row, :], shrink=0.8,
                         label="N" if row == 0 else r"$\rho$")

        def update(frame: int) -> list:
            artists = []
            for column, name in enumerate(names):
                model, result = data[name]
                index = min(frame, len(result.t) - 1)
                m = model.p.cells
                images[0][column].set_data(result.y[:model.size, index].reshape(m, m).T)
                images[1][column].set_data(result.y[model.size:, index].reshape(m, m).T)
                axes[0, column].set_title(f"{name}: N, t={result.t[index]:.1f}")
                artists.extend((images[0][column], images[1][column], axes[0, column].title))
            return artists

        frames = max(len(result.t) for _, result in data.values())
        animation = FuncAnimation(fig, update, frames=frames, interval=interval,
                                  blit=False, repeat=True)
        return fig, animation

    def show_profiles(self, data: dict, interval: int = 80) -> None:
        if plt.get_backend().lower() == "agg":
            raise RuntimeError("123")
        fig, animation = self.profile_animation(data, interval)
        try:
            plt.show(block=True)
        finally:
            plt.close(fig)

    def eigenvalues(self, data: dict) -> None:
        """Только непрерывные показатели роста первой пространственной моды."""
        fig, ax = plt.subplots(figsize=(7, 4))
        etas = [model.p.eta for model, _ in data.values()]
        eigenvalues = np.array([model.eigenvalues() for model, _ in data.values()])
        ax.plot(etas, eigenvalues[:, 0], "o-", label=r"$\gamma_+$")
        ax.plot(etas, eigenvalues[:, 1], "o-", label=r"$\gamma_-$")
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel("Показатель роста моды (1,0)")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(self.directory / "eigenvalues.png", dpi=160)
        plt.close(fig)
