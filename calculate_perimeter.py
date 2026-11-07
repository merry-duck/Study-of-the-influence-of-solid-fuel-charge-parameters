"""Расчёт и построение безразмерного периметра горения заряда."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from main import (
    burning_perimeter,
    dimensionless_geometry,
    maximum_web,
    neutral_beta,
    sector_angle,
)
from variant_parameters import (
    E0_BAR_VALUES,
    N_RAYS,
    POINTS_PER_CURVE,
    R_BAR,
)

PROJECT_DIR = Path(__file__).resolve().parent
STYLE_PATH = PROJECT_DIR / "graf_style.txt"
OUTPUT_DIR = PROJECT_DIR / "output"
OUTPUT_PATH = OUTPUT_DIR / "burning_perimeter.png"


def calculate_curve(e0_bar: float, beta: float) -> tuple[np.ndarray, np.ndarray]:
    """Рассчитать кривую безразмерного периметра горения.

    Parameters
    ----------
    e0_bar : float
        Относительная минимальная толщина свода ``2*e0/D``.
    beta : float
        Критический полуугол луча, рад.

    Returns
    -------
    e_bar : numpy.ndarray
        Безразмерная толщина сгоревшего свода ``2*e/D``.
    perimeter_bar : numpy.ndarray
        Безразмерный периметр горения ``Pi/D``.
    """
    radius_hat, r_hat = dimensionless_geometry(e0_bar, R_BAR)
    alpha = sector_angle(N_RAYS)
    e_max_bar = maximum_web(radius_hat, r_hat, alpha)
    e_bar = np.linspace(0.0, e_max_bar, POINTS_PER_CURVE)
    perimeter_bar = burning_perimeter(
        e_bar=e_bar,
        n=N_RAYS,
        beta=beta,
        e0_bar=e0_bar,
        r_hat=r_hat,
        radius_hat=radius_hat,
    )
    return e_bar, perimeter_bar


def plot_curves(beta: float) -> None:
    """Построить и сохранить семейство кривых ``Pi_bar(e_bar)``.

    Parameters
    ----------
    beta : float
        Критический полуугол луча, рад.

    Returns
    -------
    None
        Функция сохраняет график в PNG-файл.
    """
    plt.style.use(STYLE_PATH)
    figure, axes = plt.subplots()
    line_styles = ("-", "--", "-.", ":", "-", "--", "-.")

    for e0_bar, line_style in zip(E0_BAR_VALUES, line_styles, strict=True):
        e_bar, perimeter_bar = calculate_curve(float(e0_bar), beta)
        axes.plot(
            e_bar,
            perimeter_bar,
            linestyle=line_style,
            label=rf"$\bar{{e}}_0={e0_bar:.2f}$".replace(".", ","),
        )

    axes.set_xlabel(r"Толщина сгоревшего свода $\bar{e}$")
    axes.set_ylabel(r"Периметр поверхности горения $\bar{\Pi}$")
    # axes.set_title(r"Зависимость $\bar{\Pi}(\bar{e})$")
    axes.set_xlim(left=0.0)
    axes.set_ylim(bottom=0.0)
    axes.minorticks_on()
    axes.legend(ncols=2)

    OUTPUT_DIR.mkdir(exist_ok=True)
    figure.savefig(OUTPUT_PATH)
    plt.show()
    plt.close(figure)


def main() -> None:
    """Запустить расчёт для индивидуального варианта.

    Returns
    -------
    None
        Функция выводит исходные параметры и сохраняет график.
    """
    beta = neutral_beta(N_RAYS)
    plot_curves(beta)
    print(f"n = {N_RAYS}")
    print(f"r_bar = {R_BAR:.2f}")
    print(f"beta_critical = {np.degrees(beta):.6f} deg")
    print("e0_bar =", ", ".join(f"{value:.2f}" for value in E0_BAR_VALUES))
    print(f"plot = {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
