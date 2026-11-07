"""Восстановление размерных величин для заданного диаметра заряда.

Безразмерная математическая модель от диаметра не зависит. Физическое
значение диаметра задаётся только в этом модуле и используется для перевода
готовых безразмерных величин в миллиметры и квадратные миллиметры.
"""

from __future__ import annotations

from dataclasses import dataclass

from main import (
    dimensionless_geometry,
    first_stage_parameters,
    maximum_web,
    neutral_beta,
    sector_angle,
)
from variant_parameters import E0_BAR_VALUES, N_RAYS, R_BAR


# Реальный наружный диаметр заряда, мм.
DIAMETER_MM = 380.0


@dataclass(frozen=True)
class DimensionalGeometry:
    """Размерные геометрические параметры одного значения ``e0_bar``."""

    diameter_mm: float
    e0_bar: float
    r_ratio: float
    e0_mm: float
    radius_mm: float
    rounding_radius_mm: float
    e1_mm: float
    e_max_mm: float


def length_from_bar(value: float, diameter_mm: float = DIAMETER_MM) -> float:
    """Перевести безразмерную длину в миллиметры.

    Parameters
    ----------
    value : float
        Величина вида ``2*x/D``.
    diameter_mm : float, optional
        Наружный диаметр, мм.

    Returns
    -------
    float
        Размерная длина, мм.
    """
    if diameter_mm <= 0.0:
        raise ValueError("диаметр должен быть положительным")
    return diameter_mm * value / 2.0


def perimeter_from_bar(
    perimeter_bar: float, diameter_mm: float = DIAMETER_MM
) -> float:
    """Перевести безразмерный периметр в миллиметры.

    Parameters
    ----------
    perimeter_bar : float
        Периметр ``Pi_bar=Pi/D``.
    diameter_mm : float, optional
        Наружный диаметр, мм.

    Returns
    -------
    float
        Размерный периметр, мм.
    """
    if diameter_mm <= 0.0 or perimeter_bar < 0.0:
        raise ValueError("диаметр должен быть положительным, периметр — неотрицательным")
    return diameter_mm * perimeter_bar


def area_from_bar(area_bar: float, diameter_mm: float = DIAMETER_MM) -> float:
    """Перевести безразмерную площадь в квадратные миллиметры.

    Parameters
    ----------
    area_bar : float
        Площадь ``F_bar=4F/D**2``.
    diameter_mm : float, optional
        Наружный диаметр, мм.

    Returns
    -------
    float
        Размерная площадь, мм².
    """
    if diameter_mm <= 0.0 or area_bar < 0.0:
        raise ValueError("диаметр должен быть положительным, площадь — неотрицательной")
    return diameter_mm**2 * area_bar / 4.0


def calculate_dimensional_geometry(
    e0_bar: float,
    r_ratio: float = R_BAR,
    diameter_mm: float = DIAMETER_MM,
) -> DimensionalGeometry:
    """Восстановить размерную геометрию одного варианта.

    Parameters
    ----------
    e0_bar : float
        Относительная толщина свода ``2e0/D``.
    r_ratio : float, optional
        Относительное скругление ``r/R``.
    diameter_mm : float, optional
        Реальный наружный диаметр, мм.

    Returns
    -------
    DimensionalGeometry
        Набор восстановленных размерных параметров.
    """
    if diameter_mm <= 0.0:
        raise ValueError("диаметр должен быть положительным")
    radius_hat, r_hat = dimensionless_geometry(e0_bar, r_ratio)
    alpha = sector_angle(N_RAYS)
    beta = neutral_beta(N_RAYS)
    _, e1_bar = first_stage_parameters(radius_hat, r_hat, alpha, beta)
    e_max_bar = maximum_web(radius_hat, r_hat, alpha)
    return DimensionalGeometry(
        diameter_mm=diameter_mm,
        e0_bar=e0_bar,
        r_ratio=r_ratio,
        e0_mm=length_from_bar(e0_bar, diameter_mm),
        radius_mm=length_from_bar(radius_hat, diameter_mm),
        rounding_radius_mm=length_from_bar(r_hat, diameter_mm),
        e1_mm=length_from_bar(e1_bar, diameter_mm),
        e_max_mm=length_from_bar(e_max_bar, diameter_mm),
    )


def main() -> None:
    """Вывести размерные параметры индивидуального варианта.

    Returns
    -------
    None
        Функция печатает таблицу размерной геометрии.
    """
    print(f"D = {DIAMETER_MM:.3f} mm")
    print(f"n = {N_RAYS}; r/R = {R_BAR:.3f}")
    print()
    print("e0_bar    e0, mm      R, mm      r, mm     e1, mm   e_max, mm")
    for e0_bar in E0_BAR_VALUES:
        geometry = calculate_dimensional_geometry(float(e0_bar))
        print(
            f"{geometry.e0_bar:6.2f}  "
            f"{geometry.e0_mm:9.3f}  "
            f"{geometry.radius_mm:9.3f}  "
            f"{geometry.rounding_radius_mm:9.3f}  "
            f"{geometry.e1_mm:9.3f}  "
            f"{geometry.e_max_mm:10.3f}"
        )


if __name__ == "__main__":
    main()
