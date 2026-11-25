"""Начальные поля из математических выражений в config.yaml."""

import ast

import numpy as np
from scipy.ndimage import gaussian_filter
import sympy as sp


class InitialFormula:
    FUNCTIONS = {"sin": sp.sin, "cos": sp.cos, "exp": sp.exp,
                 "sqrt": sp.sqrt, "Abs": sp.Abs}
    VARIABLES = {"x", "y", "noise", "n0", "epsilon", "length", "mu", "eta",
                 "diffusion", "alpha", "beta", "sigma", "seed", "pi"}

    def __init__(self, expression: str):
        self.expression = expression
        tree = ast.parse(expression, mode="eval")
        self._check(tree)
        symbols = {name: sp.Symbol(name) for name in self.VARIABLES if name != "pi"}
        symbols["pi"] = sp.pi
        self.symbols = symbols
        self.parsed = sp.sympify(expression, locals={**symbols, **self.FUNCTIONS})

    @classmethod
    def _check(cls, node: ast.AST) -> None:
        if isinstance(node, ast.Expression):
            cls._check(node.body)
        elif isinstance(node, ast.BinOp) and isinstance(
                node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow)):
            cls._check(node.left)
            cls._check(node.right)
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            cls._check(node.operand)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id in cls.FUNCTIONS and not node.keywords:
            for argument in node.args:
                cls._check(argument)
        elif isinstance(node, ast.Name) and node.id in cls.VARIABLES:
            pass
        elif isinstance(node, ast.Constant) and type(node.value) in (int, float):
            pass
        else:
            raise ValueError("Недопустимое выражение начального поля")

    def evaluate(self, x: np.ndarray, y: np.ndarray, noise: np.ndarray,
                 parameters) -> np.ndarray:
        values = {name: getattr(parameters, name) for name in self.VARIABLES
                  if name not in ("x", "y", "noise", "pi")}
        substitutions = {self.symbols[name]: value for name, value in values.items()
                         if value is not None}
        expression = self.parsed.subs(substitutions)
        arguments = (self.symbols["x"], self.symbols["y"], self.symbols["noise"])
        if expression.free_symbols - set(arguments):
            raise ValueError("В формуле указан параметр без значения")
        function = sp.lambdify(arguments, expression, modules="numpy")
        field = np.asarray(function(x, y, noise), dtype=float)
        field = np.broadcast_to(field, x.shape).copy()
        if not np.all(np.isfinite(field)) or np.any(field < 0):
            raise ValueError("Начальная концентрация должна быть конечной и неотрицательной")
        return field


def make_noise(cells: int, sigma: float | None, seed: int | None) -> np.ndarray:
    if sigma is None or seed is None:
        return np.zeros((cells, cells))
    field = gaussian_filter(np.random.default_rng(seed).standard_normal((cells, cells)),
                            sigma=sigma, mode="reflect")
    field -= field.mean()
    scale = np.max(np.abs(field))
    if scale == 0:
        raise ValueError("Шумовое поле оказалось постоянным")
    return field / scale
