"""Термодинамический расчёт заряда по указаниям на страницах 15–16."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import (
    burning_perimeter,
    channel_area,
    degressive_residue_fraction,
    dimensionless_geometry,
    dimensionless_pobedonostsev,
    filling_coefficient,
    mean_burning_perimeter,
    neutral_beta,
    relative_area_peak,
)
from variant_parameters import (
    E0_BAR_VALUES,
    INTEGRATION_POINTS,
    N_RAYS,
    R_BAR,
)

PROJECT_DIR = Path(__file__).resolve().parent
STYLE_PATH = PROJECT_DIR / "graf_style.txt"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_PATH = OUTPUT_DIR / "thermodynamic_results.csv"
PLOT_PATH = OUTPUT_DIR / "thermodynamic_heatmap.png"


def calculate_variant(e0_bar: float, beta: float) -> dict[str, float]:
    """Выполнить термодинамический расчёт одного варианта.

    Parameters
    ----------
    e0_bar : float
        Относительная минимальная толщина свода ``2*e0/D``.
    beta : float
        Критический полуугол луча, рад.

    Returns
    -------
    dict of str to float
        Значения ``e0_bar``, ``epsilon_f``, ``kappa_bar``, ``xi_degr``
        и ``sigma_e0``.
    """
    radius_hat, r_hat = dimensionless_geometry(e0_bar, R_BAR)

    perimeter_initial = burning_perimeter(
        0.0, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    channel_area_initial = channel_area(
        0.0, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    epsilon_f = filling_coefficient(channel_area_initial)
    kappa_bar = dimensionless_pobedonostsev(perimeter_initial, epsilon_f)

    channel_area_e0 = channel_area(
        e0_bar, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    xi_degr = degressive_residue_fraction(channel_area_e0, epsilon_f)

    e_grid = np.linspace(0.0, e0_bar, INTEGRATION_POINTS)
    perimeter_grid = burning_perimeter(
        e_grid, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    perimeter_mean = mean_burning_perimeter(e_grid, perimeter_grid, e0_bar)
    sigma_e0 = relative_area_peak(perimeter_grid[-1], perimeter_mean)

    return {
        "e0_bar": float(e0_bar),
        "epsilon_f": epsilon_f,
        "kappa_bar": kappa_bar,
        "xi_degr": xi_degr,
        "sigma_e0": sigma_e0,
    }


def calculate_all_variants(beta: float) -> list[dict[str, float]]:
    """Рассчитать все значения относительной толщины свода.

    Parameters
    ----------
    beta : float
        Критический полуугол луча, рад.

    Returns
    -------
    list of dict of str to float
        Результаты для ``e0_bar`` от 0,30 до 0,60 с шагом 0,05.
    """
    return [calculate_variant(float(e0_bar), beta) for e0_bar in E0_BAR_VALUES]


def save_results(results: list[dict[str, float]]) -> None:
    """Сохранить результаты расчёта в CSV-файл.

    Parameters
    ----------
    results : list of dict of str to float
        Результаты термодинамического расчёта.

    Returns
    -------
    None
        Функция записывает таблицу на диск.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    fieldnames = ("e0_bar", "epsilon_f", "kappa_bar", "xi_degr", "sigma_e0")
    with TABLE_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames, delimiter=";")
        writer.writeheader()
        for result in results:
            writer.writerow({key: f"{result[key]:.8f}" for key in fieldnames})


def plot_heatmap(results: list[dict[str, float]]) -> None:
    """Построить тепловую диаграмму термодинамического расчёта.

    Parameters
    ----------
    results : list of dict of str to float
        Результаты термодинамического расчёта.

    Returns
    -------
    None
        Функция сохраняет тепловую диаграмму в PNG-файл.
    """
    plt.style.use(STYLE_PATH)
    kappa = np.asarray([result["kappa_bar"] for result in results])
    epsilon = np.asarray([result["epsilon_f"] for result in results])
    sigma = np.asarray([result["sigma_e0"] for result in results])

    figure, axes = plt.subplots()
    points = axes.scatter(
        kappa,
        epsilon,
        c=sigma,
        cmap="plasma",
        s=90,
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )

    for index, result in enumerate(results):
        label = rf"$\bar{{e}}_0={result['e0_bar']:.2f}$".replace(".", ",")
        is_last_point = index == len(results) - 1
        axes.annotate(
            label,
            (result["kappa_bar"], result["epsilon_f"]),
            xytext=(-4, -18) if is_last_point else (4, 8),
            textcoords="offset points",
            fontsize=10,
            horizontalalignment="right" if is_last_point else "left",
        )

    colorbar = figure.colorbar(points, ax=axes)
    colorbar.set_label(r"$\sigma_{e_0}$")
    axes.set_xlabel(r"Безразмерный параметр Победоносцева $\bar{\kappa}$")
    axes.set_ylabel(r"Коэффициент заполнения $\varepsilon_f$")
    # axes.set_title("Тепловая диаграмма")
    axes.minorticks_on()

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(PLOT_PATH)
    plt.show()
    plt.close(figure)


def print_results(results: list[dict[str, float]], beta: float) -> None:
    """Вывести исходные данные и расчётную таблицу в консоль.

    Parameters
    ----------
    results : list of dict of str to float
        Результаты термодинамического расчёта.
    beta : float
        Критический полуугол луча, рад.

    Returns
    -------
    None
        Функция печатает таблицу в стандартный поток вывода.
    """
    print(f"n = {N_RAYS}")
    print(f"r_bar = {R_BAR:.2f}")
    print(f"beta_critical = {np.degrees(beta):.6f} deg")
    print()
    print("e0_bar   epsilon_f   kappa_bar   xi_degr    sigma_e0")
    for result in results:
        print(
            f"{result['e0_bar']:6.2f}   "
            f"{result['epsilon_f']:9.6f}   "
            f"{result['kappa_bar']:9.6f}   "
            f"{result['xi_degr']:8.6f}   "
            f"{result['sigma_e0']:8.6f}"
        )


def main() -> None:
    """Запустить термодинамический расчёт индивидуального варианта.

    Returns
    -------
    None
        Функция сохраняет расчётную таблицу и тепловую диаграмму.
    """
    beta = neutral_beta(N_RAYS)
    results = calculate_all_variants(beta)
    save_results(results)
    plot_heatmap(results)
    print_results(results, beta)
    print(f"\ntable = {TABLE_PATH}")
    print(f"plot = {PLOT_PATH}")


if __name__ == "__main__":
    main()
