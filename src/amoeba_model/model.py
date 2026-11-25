"""Двумерная конечно-объёмная дискретизация модели хемотаксиса."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix

from .initial import InitialFormula, make_noise


@dataclass(frozen=True)
class Parameters:
    length: float = 10.0
    n0: float = 1.0
    epsilon: float = 0.01
    mu: float = 0.2
    eta: float = 0.1
    diffusion: float = 1.0
    alpha: float = 1.0
    beta: float = 1.0
    cells: int = 20  # ячеек вдоль каждой стороны квадрата
    initial_n: str = "n0*(1 + epsilon*(cos(pi*x/length) + cos(pi*y/length))/2)"
    initial_rho: str = "alpha*n0/beta"
    sigma: float | None = None
    seed: int | None = None

    def __post_init__(self) -> None:
        for name in ("length", "n0", "mu", "eta", "diffusion", "alpha", "beta"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} должен быть положительным конечным числом")
        if not np.isfinite(self.epsilon):
            raise ValueError("epsilon должен быть конечным числом")
        if not isinstance(self.cells, int) or self.cells < 3:
            raise ValueError("Нужно не менее трёх ячеек по каждой оси")
        if self.sigma is not None and (not np.isfinite(self.sigma) or self.sigma <= 0):
            raise ValueError("sigma должен быть положительным")
        if self.seed is not None and not isinstance(self.seed, int):
            raise ValueError("seed должен быть целым")


class AmoebaModel:
    """Хранит 2*m*m неизвестных в порядке [N.flatten(), rho.flatten()]."""

    def __init__(self, parameters: Parameters):
        self.p = parameters
        self.dx = parameters.length / parameters.cells
        self.x = (np.arange(parameters.cells) + 0.5) * self.dx
        self.size = parameters.cells ** 2
        self.initial_n_formula = InitialFormula(parameters.initial_n)
        self.initial_rho_formula = InitialFormula(parameters.initial_rho)
        if any(self.initial_n_formula.symbols["noise"] in formula.parsed.free_symbols
               for formula in (self.initial_n_formula, self.initial_rho_formula)):
            if parameters.sigma is None or parameters.seed is None:
                raise ValueError("Для noise нужны sigma и seed")
        self._initial_state = self._build_initial_state()
        self.equilibrium_n = float(np.mean(self._initial_state[:self.size]))
        self.equilibrium_rho = parameters.alpha * self.equilibrium_n / parameters.beta
        self.jac_sparsity = self._jacobian_sparsity()

    def _build_initial_state(self) -> NDArray[np.float64]:
        p = self.p
        x, y = np.meshgrid(self.x, self.x, indexing="ij")
        noise = make_noise(p.cells, p.sigma, p.seed)
        n = self.initial_n_formula.evaluate(x, y, noise, p)
        rho = self.initial_rho_formula.evaluate(x, y, noise, p)
        return np.concatenate((n.ravel(), rho.ravel()))

    def initial_state(self) -> NDArray[np.float64]:
        return self._initial_state.copy()

    def equilibrium_state(self) -> NDArray[np.float64]:
        return np.concatenate((np.full(self.size, self.equilibrium_n),
                               np.full(self.size, self.equilibrium_rho)))

    def rhs(self, _time: float, state: NDArray[np.float64]) -> NDArray[np.float64]:
        p, h, m = self.p, self.dx, self.p.cells
        n = state[:self.size].reshape(m, m)
        rho = state[self.size:].reshape(m, m)

        vx = p.eta * np.diff(rho, axis=0) / h
        vy = p.eta * np.diff(rho, axis=1) / h
        jx = np.zeros((m + 1, m))
        jy = np.zeros((m, m + 1))
        jx[1:-1, :] = (-p.mu * np.diff(n, axis=0) / h
                        + np.maximum(vx, 0) * n[:-1, :]
                        + np.minimum(vx, 0) * n[1:, :])
        jy[:, 1:-1] = (-p.mu * np.diff(n, axis=1) / h
                        + np.maximum(vy, 0) * n[:, :-1]
                        + np.minimum(vy, 0) * n[:, 1:])
        dn = -(np.diff(jx, axis=0) + np.diff(jy, axis=1)) / h

        fx = np.zeros((m + 1, m))
        fy = np.zeros((m, m + 1))
        fx[1:-1, :] = -p.diffusion * np.diff(rho, axis=0) / h
        fy[:, 1:-1] = -p.diffusion * np.diff(rho, axis=1) / h
        drho = -(np.diff(fx, axis=0) + np.diff(fy, axis=1)) / h
        drho += p.alpha * n - p.beta * rho
        return np.concatenate((dn.ravel(), drho.ravel()))

    def _jacobian_sparsity(self):
        """оптимизация разреженного численного якобиана BDF."""
        m, s = self.p.cells, self.size
        pattern = lil_matrix((2 * s, 2 * s), dtype=bool)
        for i in range(m):
            for j in range(m):
                cell = i * m + j
                neighbors = [cell]
                if i > 0:
                    neighbors.append(cell - m)
                if i + 1 < m:
                    neighbors.append(cell + m)
                if j > 0:
                    neighbors.append(cell - 1)
                if j + 1 < m:
                    neighbors.append(cell + 1)
                pattern[cell, neighbors] = True
                pattern[cell, [s + k for k in neighbors]] = True
                pattern[s + cell, cell] = True
                pattern[s + cell, [s + k for k in neighbors]] = True
        return pattern.tocsr()

    def solve(self, t_end: float, samples: int = 121,
              initial: NDArray[np.float64] | None = None):
        if t_end <= 0 or samples < 2:
            raise ValueError("Нужны t_end>0 и samples>=2")
        result = solve_ivp(self.rhs, (0, t_end), self.initial_state() if initial is None else initial,
                           method="BDF", jac_sparsity=self.jac_sparsity,
                           t_eval=np.linspace(0, t_end, samples), rtol=1e-8, atol=1e-10)
        if not result.success:
            raise RuntimeError(result.message)
        return result

    def eigenvalues(self, mode: tuple[int, int] = (1, 0)) -> NDArray[np.float64]:
        """Собственные значения непрерывной модели для моды (n_x, n_y)."""
        if mode == (0, 0) or any(n < 0 for n in mode):
            raise ValueError("Нужна ненулевая мода с неотрицательными номерами")
        p = self.p
        q = float((np.pi / p.length) ** 2 * (mode[0] ** 2 + mode[1] ** 2))
        matrix = np.array([[-p.mu * q, p.eta * self.equilibrium_n * q],
                           [p.alpha, -(p.diffusion * q + p.beta)]])
        return np.sort(np.linalg.eigvals(matrix).real)[::-1]
