"""Безразмерная математическая модель заряда со звездообразным каналом."""

from __future__ import annotations

import numpy as np


def sector_angle(n: int) -> float:
    """Вычислить угол сектора звезды.

    Parameters
    ----------
    n : int
        Число лучей звезды.

    Returns
    -------
    float
        Угол ``alpha = pi/n``, рад.
    """
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)):
        raise ValueError("n должно быть целым числом")
    if not 5 <= n <= 11:
        raise ValueError("по методике n должно находиться в диапазоне [5, 11]")
    return float(np.pi / n)


def neutral_beta(n: int, tolerance: float = 1e-12) -> float:
    """Найти критический полуугол нейтрального горения.

    Parameters
    ----------
    n : int
        Число лучей звезды.
    tolerance : float, optional
        Абсолютная точность решения по углу, рад.

    Returns
    -------
    float
        Критический полуугол ``beta``, рад.
    """
    alpha = sector_angle(n)
    if not np.isfinite(tolerance) or tolerance <= 0.0:
        raise ValueError("tolerance должна быть положительной")

    def equation(beta: float) -> float:
        """Вычислить невязку уравнения нейтрального угла.

        Parameters
        ----------
        beta : float
            Проверяемый полуугол, рад.

        Returns
        -------
        float
            Невязка уравнения.
        """
        return np.pi / 2.0 + alpha - beta - 1.0 / np.tan(beta)

    left, right = alpha, np.pi / 2.0 - np.finfo(float).eps
    f_left = equation(left)
    if f_left * equation(right) > 0.0:
        raise ValueError("корень beta не заключён на допустимом интервале")
    while right - left > tolerance:
        middle = (left + right) / 2.0
        f_middle = equation(middle)
        if f_left * f_middle <= 0.0:
            right = middle
        else:
            left, f_left = middle, f_middle
    return float((left + right) / 2.0)


def dimensionless_geometry(e0_bar: float, r_ratio: float) -> tuple[float, float]:
    """Вычислить безразмерные радиусы геометрии.

    Parameters
    ----------
    e0_bar : float
        Относительная толщина свода ``2*e0/D``.
    r_ratio : float
        Относительное скругление ``r/R``.

    Returns
    -------
    radius_hat : float
        Безразмерный радиус ``2R/D``.
    r_hat : float
        Безразмерный радиус скругления ``2r/D``.
    """
    values = np.asarray([e0_bar, r_ratio], dtype=float)
    if np.any(~np.isfinite(values)):
        raise ValueError("геометрические параметры должны быть конечными")
    if not 0.0 < e0_bar < 1.0 or r_ratio < 0.0:
        raise ValueError("требуется 0 < e0_bar < 1 и r_ratio >= 0")
    radius_hat = (1.0 - e0_bar) / (1.0 + r_ratio)
    r_hat = r_ratio * radius_hat
    return float(radius_hat), float(r_hat)


def first_stage_parameters(
    radius_hat: float, r_hat: float, alpha: float, beta: float
) -> tuple[float, float]:
    """Вычислить параметры первой стадии.

    Parameters
    ----------
    radius_hat : float
        Безразмерный радиус ``2R/D``.
    r_hat : float
        Безразмерный радиус скругления ``2r/D``.
    alpha : float
        Угол сектора, рад.
    beta : float
        Полуугол луча, рад.

    Returns
    -------
    gamma1 : float
        Постоянный угол первой стадии, рад.
    e1_bar : float
        Безразмерная граница первой стадии.
    """
    if radius_hat <= 0.0 or r_hat < 0.0 or not 0.0 < alpha <= beta < np.pi / 2.0:
        raise ValueError("недопустимые параметры первой стадии")
    gamma1 = np.pi / 2.0 + alpha - beta
    e1_bar = radius_hat * np.sin(alpha) / np.cos(beta) - r_hat
    return float(gamma1), float(e1_bar)


def gamma_angle(e_bar, radius_hat: float, r_hat: float, alpha: float):
    """Вычислить угол второй и третьей стадий.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Безразмерная толщина сгоревшего свода.
    radius_hat, r_hat : float
        Безразмерные радиусы ``2R/D`` и ``2r/D``.
    alpha : float
        Угол сектора, рад.

    Returns
    -------
    float or numpy.ndarray
        Угол ``gamma(e_bar)``, рад.
    """
    e_array = np.asarray(e_bar, dtype=float)
    argument = radius_hat * np.sin(alpha) / (r_hat + e_array)
    if np.any(e_array < 0.0) or np.any(np.abs(argument) > 1.0 + 1e-12):
        raise ValueError("аргумент gamma находится вне области определения")
    result = alpha + np.arcsin(np.clip(argument, -1.0, 1.0))
    return float(result) if e_array.ndim == 0 else result


def delta_angle(e_bar, radius_hat: float, r_hat: float):
    """Вычислить угол отсечения дуги на третьей стадии.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Безразмерная толщина сгоревшего свода.
    radius_hat, r_hat : float
        Безразмерные радиусы ``2R/D`` и ``2r/D``.

    Returns
    -------
    float or numpy.ndarray
        Угол ``delta(e_bar)``, рад.
    """
    e_array = np.asarray(e_bar, dtype=float)
    current_radius = r_hat + e_array
    argument = (radius_hat**2 + current_radius**2 - 1.0) / (
        2.0 * radius_hat * current_radius
    )
    if np.any(np.abs(argument) > 1.0 + 1e-12):
        raise ValueError("аргумент delta находится вне области определения")
    result = np.pi - np.arccos(np.clip(argument, -1.0, 1.0))
    return float(result) if e_array.ndim == 0 else result


def phi_angle(e_bar, radius_hat: float, r_hat: float):
    """Вычислить вспомогательный угол площади канала.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Безразмерная толщина сгоревшего свода.
    radius_hat, r_hat : float
        Безразмерные радиусы ``2R/D`` и ``2r/D``.

    Returns
    -------
    float or numpy.ndarray
        Угол ``phi(e_bar)``, рад.
    """
    e_array = np.asarray(e_bar, dtype=float)
    argument = (radius_hat**2 + 1.0 - (r_hat + e_array) ** 2) / (
        2.0 * radius_hat
    )
    if np.any(np.abs(argument) > 1.0 + 1e-12):
        raise ValueError("аргумент phi находится вне области определения")
    result = np.arccos(np.clip(argument, -1.0, 1.0))
    return float(result) if e_array.ndim == 0 else result


def maximum_web(radius_hat: float, r_hat: float, alpha: float) -> float:
    """Вычислить максимальную безразмерную толщину свода.

    Parameters
    ----------
    radius_hat, r_hat : float
        Безразмерные радиусы ``2R/D`` и ``2r/D``.
    alpha : float
        Угол сектора, рад.

    Returns
    -------
    float
        Толщина ``e_max_bar``.
    """
    return float(
        np.sqrt(radius_hat**2 + 1.0 - 2.0 * radius_hat * np.cos(alpha)) - r_hat
    )


def burning_perimeter(e_bar, n, beta, e0_bar, r_hat, radius_hat):
    """Вычислить безразмерный периметр горения.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Безразмерная толщина сгоревшего свода.
    n : int
        Число лучей.
    beta : float
        Полуугол луча, рад.
    e0_bar, r_hat, radius_hat : float
        Безразмерные параметры геометрии.

    Returns
    -------
    float or numpy.ndarray
        Периметр ``Pi_bar=Pi/D``.
    """
    e_array, alpha, gamma1, e1_bar, e_max_bar = _model_domain(
        e_bar, n, beta, e0_bar, r_hat, radius_hat
    )
    result = np.empty_like(e_array)
    first = e_array <= e1_bar
    second = (e_array > e1_bar) & (e_array <= e0_bar)
    third = e_array > e0_bar
    if np.any(first):
        ef = e_array[first]
        b_hat = radius_hat * np.sin(alpha) / np.sin(beta) - (r_hat + ef) / np.tan(beta)
        result[first] = n * (gamma1 * (r_hat + ef) + b_hat)
    if np.any(second):
        es = e_array[second]
        result[second] = n * np.asarray(gamma_angle(es, radius_hat, r_hat, alpha)) * (r_hat + es)
    if np.any(third):
        et = e_array[third]
        gamma = np.asarray(gamma_angle(et, radius_hat, r_hat, alpha))
        delta = np.asarray(delta_angle(et, radius_hat, r_hat))
        result[third] = n * (gamma - delta) * (r_hat + et)
    if np.any(e_array > e_max_bar + 1e-12):
        raise ValueError("e_bar превышает e_max_bar")
    return float(result) if e_array.ndim == 0 else result


def channel_area(e_bar, n, beta, e0_bar, r_hat, radius_hat):
    """Вычислить безразмерную площадь канала.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Безразмерная толщина сгоревшего свода.
    n : int
        Число лучей.
    beta : float
        Полуугол луча, рад.
    e0_bar, r_hat, radius_hat : float
        Безразмерные параметры геометрии.

    Returns
    -------
    float or numpy.ndarray
        Площадь ``F_bar=4*F_channel/D**2``.
    """
    e_array, alpha, gamma1, e1_bar, e_max_bar = _model_domain(
        e_bar, n, beta, e0_bar, r_hat, radius_hat
    )
    result = np.empty_like(e_array)
    first = e_array <= e1_bar
    second = (e_array > e1_bar) & (e_array <= e0_bar)
    third = e_array > e0_bar
    if np.any(first):
        ef = e_array[first]
        current_radius = r_hat + ef
        b_hat = radius_hat * np.sin(alpha) / np.sin(beta) - current_radius / np.tan(beta)
        result[first] = n * (
            gamma1 * current_radius**2
            + b_hat * (current_radius + radius_hat * np.cos(gamma1))
            + radius_hat * current_radius * np.sin(gamma1)
        )
    if np.any(second):
        es = e_array[second]
        current_radius = r_hat + es
        gamma = np.asarray(gamma_angle(es, radius_hat, r_hat, alpha))
        result[second] = n * (
            gamma * current_radius**2 + radius_hat * current_radius * np.sin(gamma)
        )
    if np.any(third):
        et = e_array[third]
        current_radius = r_hat + et
        gamma = np.asarray(gamma_angle(et, radius_hat, r_hat, alpha))
        delta = np.asarray(delta_angle(et, radius_hat, r_hat))
        phi = np.asarray(phi_angle(et, radius_hat, r_hat))
        result[third] = n * (
            (gamma - delta) * current_radius**2
            + radius_hat * current_radius * np.sin(gamma)
            + phi - radius_hat * np.sin(phi)
        )
    if np.any(e_array > e_max_bar + 1e-12):
        raise ValueError("e_bar превышает e_max_bar")
    return float(result) if e_array.ndim == 0 else result


def filling_coefficient(channel_area_bar_initial: float) -> float:
    """Вычислить коэффициент заполнения сечения.

    Parameters
    ----------
    channel_area_bar_initial : float
        Начальная безразмерная площадь канала.

    Returns
    -------
    float
        Коэффициент ``epsilon_f=1-F_bar(0)/pi``.
    """
    epsilon_f = 1.0 - channel_area_bar_initial / np.pi
    if not 0.0 < epsilon_f < 1.0:
        raise ValueError("получено недопустимое значение epsilon_f")
    return float(epsilon_f)


def dimensionless_pobedonostsev(perimeter_bar_initial: float, epsilon_f: float) -> float:
    """Вычислить безразмерный параметр Победоносцева.

    Parameters
    ----------
    perimeter_bar_initial : float
        Начальный безразмерный периметр.
    epsilon_f : float
        Коэффициент заполнения сечения.

    Returns
    -------
    float
        Безразмерный параметр Победоносцева.
    """
    if perimeter_bar_initial <= 0.0 or not 0.0 < epsilon_f < 1.0:
        raise ValueError("недопустимые параметры Победоносцева")
    return float(perimeter_bar_initial / (epsilon_f * (1.0 - epsilon_f)))


def degressive_residue_fraction(channel_area_bar_at_e0: float, epsilon_f: float) -> float:
    """Вычислить массовую долю дегрессивных остатков.

    Parameters
    ----------
    channel_area_bar_at_e0 : float
        Безразмерная площадь канала при ``e_bar=e0_bar``.
    epsilon_f : float
        Коэффициент заполнения сечения.

    Returns
    -------
    float
        Массовая доля дегрессивных остатков.
    """
    if channel_area_bar_at_e0 < 0.0 or not 0.0 < epsilon_f < 1.0:
        raise ValueError("недопустимые параметры доли остатков")
    return float((1.0 - channel_area_bar_at_e0 / np.pi) / epsilon_f)


def mean_burning_perimeter(e_bar, perimeter_bar, e0_bar: float) -> float:
    """Вычислить средний безразмерный периметр.

    Parameters
    ----------
    e_bar : numpy.ndarray
        Возрастающая сетка от нуля до ``e0_bar``.
    perimeter_bar : numpy.ndarray
        Значения безразмерного периметра на сетке.
    e0_bar : float
        Верхняя граница интервала усреднения.

    Returns
    -------
    float
        Средний безразмерный периметр.
    """
    e_array = np.asarray(e_bar, dtype=float)
    perimeter_array = np.asarray(perimeter_bar, dtype=float)
    if e_array.ndim != 1 or e_array.shape != perimeter_array.shape or e_array.size < 2:
        raise ValueError("массивы должны быть одномерными и иметь одну длину")
    if np.any(np.diff(e_array) <= 0.0) or not np.isclose(e_array[0], 0.0) or not np.isclose(e_array[-1], e0_bar):
        raise ValueError("сетка должна строго возрастать от 0 до e0_bar")
    return float(np.trapezoid(perimeter_array, e_array) / e0_bar)


def relative_area_peak(perimeter_at_e0: float, mean_perimeter: float) -> float:
    """Вычислить относительный всплеск площади горения.

    Parameters
    ----------
    perimeter_at_e0 : float
        Безразмерный периметр при ``e_bar=e0_bar``.
    mean_perimeter : float
        Средний безразмерный периметр.

    Returns
    -------
    float
        Отношение ``sigma_e0``.
    """
    if perimeter_at_e0 <= 0.0 or mean_perimeter <= 0.0:
        raise ValueError("периметры должны быть положительными")
    return float(perimeter_at_e0 / mean_perimeter)


def _model_domain(e_bar, n, beta, e0_bar, r_hat, radius_hat):
    """Проверить область применимости трёхстадийной модели.

    Parameters
    ----------
    e_bar : float or numpy.ndarray
        Проверяемые значения толщины свода.
    n : int
        Число лучей.
    beta : float
        Полуугол луча, рад.
    e0_bar, r_hat, radius_hat : float
        Безразмерные параметры геометрии.

    Returns
    -------
    tuple
        Массив толщин, ``alpha``, ``gamma1``, ``e1_bar`` и ``e_max_bar``.
    """
    e_array = np.asarray(e_bar, dtype=float)
    alpha = sector_angle(n)
    gamma1, e1_bar = first_stage_parameters(radius_hat, r_hat, alpha, beta)
    e_max_bar = maximum_web(radius_hat, r_hat, alpha)
    if np.any(~np.isfinite(e_array)) or np.any(e_array < 0.0):
        raise ValueError("e_bar должно быть конечным и неотрицательным")
    if not np.isclose(radius_hat + r_hat + e0_bar, 1.0, atol=1e-10):
        raise ValueError("геометрия должна удовлетворять R_hat+r_hat+e0_bar=1")
    if e1_bar < -1e-10 or e1_bar > e0_bar + 1e-10 or e_max_bar < e0_bar - 1e-10:
        raise ValueError("нарушен порядок стадий 0 <= e1_bar <= e0_bar <= e_max_bar")
    return e_array, alpha, gamma1, max(0.0, e1_bar), e_max_bar
