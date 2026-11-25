"""Запуск одного из трёх опытов, заданных в config.yaml."""

import json
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import yaml

from .model import AmoebaModel, Parameters
from .plots import Plotter

CONFIG = Path("config.yaml")


def load_experiment(number: int, path: Path = CONFIG) -> tuple[dict, dict]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    experiment = config["experiments"][number]
    return config["common"], experiment


def run_case(parameters: Parameters, t_end: float, samples: int):
    model = AmoebaModel(parameters)
    result = model.solve(t_end, samples)
    return model, result, {
        "eta": parameters.eta,
        "equilibrium_n": model.equilibrium_n,
        "equilibrium_rho": model.equilibrium_rho,
        "continuous_eigenvalues_mode_10": model.eigenvalues().tolist(),
        "continuous_eigenvalues_mode_01": model.eigenvalues((0, 1)).tolist(),
        "minimum_n": float(np.min(result.y[:model.size])),
        "minimum_rho": float(np.min(result.y[model.size:])),
        "nfev": result.nfev,
    }


def run_experiment(number: int, *, show_animation: bool = True,
                   config_path: Path = CONFIG, output_root: Path = Path("figures")) -> dict:
    common, experiment = load_experiment(number, config_path)
    parameter_names = {field.name for field in fields(Parameters)}
    parameter_values = {key: value for key, value in {**common, **experiment}.items()
                        if key in parameter_names and key != "eta"}
    t_end = experiment["t_end"]
    samples = experiment.get("samples", 121)
    cases = {}
    report = {"experiment": number, "settings": {**common, **experiment},
              "first_mode": [1, 0], "cases": []}
    for eta in experiment["eta"]:
        parameters = Parameters(**parameter_values, eta=eta)
        model, result, metrics = run_case(parameters, t_end, samples)
        cases[f"η={eta:g}"] = (model, result)
        report["cases"].append(metrics)
    plotter = Plotter(output_root / f"experiment_{number}")
    plotter.eigenvalues(cases)
    (plotter.directory / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if show_animation:
        plotter.show_profiles(cases, experiment.get("interval", 80))
    return report


def main() -> None:
    if len(sys.argv) != 2:
        raise ValueError("Укажите номер эксперимента")
    run_experiment(int(sys.argv[1]))


if __name__ == "__main__":
    main()
