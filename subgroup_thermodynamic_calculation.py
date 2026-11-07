"""Термодинамический расчёт семи вариантов подгруппы."""

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
    SELECTED_E0_BAR,
    SELECTED_VARIANT,
    SUBGROUP_VARIANTS,
)

PROJECT_DIR = Path(__file__).resolve().parent
STYLE_PATH = PROJECT_DIR / "graf_style.txt"
OUTPUT_DIR = PROJECT_DIR / "output"
TABLE_PATH = OUTPUT_DIR / "subgroup_thermodynamic_results.csv"
PLOT_PATH = OUTPUT_DIR / "subgroup_thermodynamic_heatmap.png"


def calculate_point(
    variant: int,
    e0_bar: float,
    r_bar: float,
    beta: float,
    beta_multiplier: float,
) -> dict[str, float | int]:
    """Рассчитать одну точку варианта подгруппы.

    Parameters
    ----------
    variant : int
        Номер варианта в подгруппе.
    e0_bar : float
        Относительная минимальная толщина свода ``2*e0/D``.
    r_bar : float
        Относительный радиус скругления ``r/R``.
    beta : float
        Полуугол луча для варианта, рад.
    beta_multiplier : float
        Множитель критического угла ``beta*``.

    Returns
    -------
    dict of str to float or int
        Геометрические параметры и результаты расчёта точки.
    """
    radius_hat, r_hat = dimensionless_geometry(e0_bar, r_bar)
    perimeter_initial = burning_perimeter(
        0.0, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    area_initial = channel_area(
        0.0, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    epsilon_f = filling_coefficient(area_initial)
    kappa_bar = dimensionless_pobedonostsev(perimeter_initial, epsilon_f)
    area_e0 = channel_area(
        e0_bar, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    xi_degr = degressive_residue_fraction(area_e0, epsilon_f)

    e_grid = np.linspace(0.0, e0_bar, INTEGRATION_POINTS)
    perimeter = burning_perimeter(
        e_grid, N_RAYS, beta, e0_bar, r_hat, radius_hat
    )
    mean_perimeter = mean_burning_perimeter(e_grid, perimeter, e0_bar)
    sigma_e0 = relative_area_peak(perimeter[-1], mean_perimeter)

    return {
        "variant": variant,
        "r_bar": r_bar,
        "beta_multiplier": beta_multiplier,
        "beta_deg": float(np.degrees(beta)),
        "e0_bar": e0_bar,
        "epsilon_f": epsilon_f,
        "kappa_bar": kappa_bar,
        "xi_degr": xi_degr,
        "sigma_e0": sigma_e0,
    }


def calculate_subgroup() -> dict[int, list[dict[str, float | int]]]:
    """Выполнить расчёт всех вариантов подгруппы.

    Returns
    -------
    dict of int to list of dict
        Результаты семи вариантов для всех значений ``e0_bar``.
    """
    beta_critical = neutral_beta(N_RAYS)
    results = {}
    for variant, (r_bar, beta_multiplier) in SUBGROUP_VARIANTS.items():
        beta = beta_multiplier * beta_critical
        results[variant] = [
            calculate_point(
                variant,
                float(e0_bar),
                r_bar,
                beta,
                beta_multiplier,
            )
            for e0_bar in E0_BAR_VALUES
        ]
    return results


def save_results(results: dict[int, list[dict[str, float | int]]]) -> None:
    """Сохранить сводную таблицу подгруппы в CSV-файл.

    Parameters
    ----------
    results : dict of int to list of dict
        Результаты расчёта подгруппы.

    Returns
    -------
    None
        Функция записывает CSV-файл.
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    fields = (
        "variant",
        "r_bar",
        "beta_multiplier",
        "beta_deg",
        "e0_bar",
        "epsilon_f",
        "kappa_bar",
        "xi_degr",
        "sigma_e0",
    )
    with TABLE_PATH.open("w", encoding="utf-8-sig", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields, delimiter=";")
        writer.writeheader()
        for variant_results in results.values():
            for result in variant_results:
                row = {
                    key: result[key] if key == "variant" else f"{result[key]:.8f}"
                    for key in fields
                }
                writer.writerow(row)


def plot_subgroup_heatmap(
    results: dict[int, list[dict[str, float | int]]]
) -> None:
    """Построить тепловую диаграмму всей подгруппы.

    Parameters
    ----------
    results : dict of int to list of dict
        Результаты расчёта подгруппы.

    Returns
    -------
    None
        Функция сохраняет диаграмму в PNG-файл.
    """
    plt.style.use(STYLE_PATH)
    figure, axes = plt.subplots()
    colors = (
        "#ff0000",
        "#1f77b4",
        "#1b5e20",
        "#4a148c",
        "#000000",
        "#ff7f0e",
        "#17becf",
    )
    line_styles = ("-", "--", "-.", ":", "-", "--", "-.")

    all_sigma = np.asarray(
        [point["sigma_e0"] for curve in results.values() for point in curve],
        dtype=float,
    )
    normalizer = plt.Normalize(all_sigma.min(), all_sigma.max())

    for index, (variant, curve) in enumerate(results.items()):
        kappa = np.asarray([point["kappa_bar"] for point in curve], dtype=float)
        epsilon = np.asarray([point["epsilon_f"] for point in curve], dtype=float)
        sigma = np.asarray([point["sigma_e0"] for point in curve], dtype=float)
        selected = variant == SELECTED_VARIANT
        r_bar, beta_multiplier = SUBGROUP_VARIANTS[variant]
        r_label = f"{r_bar:.2f}".replace(".", ",")
        beta_label = f"{beta_multiplier:.1f}".replace(".", ",")
        beta_expression = (
            r"\beta^*"
            if np.isclose(beta_multiplier, 1.0)
            else rf"{beta_label}\beta^*"
        )
        legend_label = (
            rf"Вариант №{variant} "
            rf"($\bar{{r}}={r_label};\ \beta={beta_expression}$)"
        )

        axes.plot(
            kappa,
            epsilon,
            color=colors[index],
            linestyle=line_styles[index],
            linewidth=2.0 if selected else 1.3,
            alpha=1.0 if selected else 0.75,
            label=legend_label,
            zorder=3 if selected else 1,
        )
        axes.scatter(
            kappa,
            epsilon,
            c=sigma,
            cmap="plasma",
            norm=normalizer,
            marker="D" if selected else "o",
            s=55 if selected else 38,
            edgecolors=colors[index],
            linewidths=1.2 if selected else 0.8,
            zorder=4 if selected else 2,
        )
        if selected:
            selected_point = next(
                point
                for point in curve
                if np.isclose(point["e0_bar"], SELECTED_E0_BAR)
            )
            axes.scatter(
                selected_point["kappa_bar"],
                selected_point["epsilon_f"],
                marker="*",
                s=230,
                color="#7b1fa2",
                edgecolors="black",
                linewidths=0.9,
                zorder=6,
            )

    scalar_mappable = plt.cm.ScalarMappable(norm=normalizer, cmap="plasma")
    colorbar = figure.colorbar(scalar_mappable, ax=axes)
    colorbar.set_label(r"$\sigma_{e_0}$")
    axes.set_xlabel(r"Безразмерный параметр Победоносцева $\bar{\kappa}$")
    axes.set_ylabel(r"Коэффициент заполнения $\varepsilon_f$")
    # axes.set_title("Тепловая диаграмма подгруппы")
    axes.minorticks_on()
    legend = axes.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncols=2,
        fontsize=10,
    )
    legend.get_texts()[SELECTED_VARIANT - 1].set_fontweight("bold")

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(PLOT_PATH)
    plt.show()
    plt.close(figure)


def print_summary(results: dict[int, list[dict[str, float | int]]]) -> None:
    """Вывести краткие сведения о рассчитанных вариантах.

    Parameters
    ----------
    results : dict of int to list of dict
        Результаты расчёта подгруппы.

    Returns
    -------
    None
        Функция выводит параметры вариантов в консоль.
    """
    beta_critical_deg = np.degrees(neutral_beta(N_RAYS))
    print(f"n = {N_RAYS}; beta_critical = {beta_critical_deg:.6f} deg")
    for variant, curve in results.items():
        first = curve[0]
        selected = " <- выбранный" if variant == SELECTED_VARIANT else ""
        print(
            f"Вариант {variant}: r_bar={first['r_bar']:.2f}; "
            f"beta={first['beta_multiplier']:.1f}*beta_critical; "
            f"точек={len(curve)}{selected}"
        )


def main() -> None:
    """Запустить расчёт и построение диаграммы подгруппы.

    Returns
    -------
    None
        Функция сохраняет таблицу и тепловую диаграмму.
    """
    results = calculate_subgroup()
    save_results(results)
    plot_subgroup_heatmap(results)
    print_summary(results)
    print(f"\ntable = {TABLE_PATH}")
    print(f"plot = {PLOT_PATH}")


if __name__ == "__main__":
    main()
