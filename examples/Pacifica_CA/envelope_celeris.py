"""
Aplicación de una envolvente espacial B(y) a una señal generada con la tabla
de Celeris [|hat_eta|, T, theta, phi], mediante convolución en Fourier.


Por Jose Galaz
asistido con Claude

Convenciones
------------
Tabla Celeris (formato coseno):
    eta_0(y, t) = sum_i |hat_eta|_i * cos(lambda_i * y - omega_i * t + phi_i)
con
    omega_i  = 2*pi / T_i
    |kappa_i| obtenido por dispersión lineal a profundidad h
    lambda_i = |kappa_i| * sin(theta_i)

Fourier 2D:
    eta(y, t) = (1 / (2 pi)^2) * iint hat_eta(lambda, omega)
                                 * exp(i (lambda y - omega t)) dlambda domega

Envolvente B(y) (independiente de t):
    eta_1(y, t) = eta_0(y, t) * B(y)
    => hat_eta_1(lambda, omega) = (1/(2 pi)) hat_B(lambda) *_lambda hat_eta_0
    => para cada fila i, lambda_i se "ensancha" a una distribución
       proporcional a hat_B(lambda - lambda_i), manteniendo omega_i fijo.

Resultado: una nueva tabla [|hat_eta|', T', theta', phi'] en el mismo formato.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Callable

GRAV = 9.81


# -----------------------------------------------------------------------------
# Dispersión lineal: T -> |kappa|
# -----------------------------------------------------------------------------

def wavenumber_from_period(T: float, h: float, g: float = GRAV,
                           tol: float = 1e-12, itmax: int = 200) -> float:
    """Resuelve omega^2 = g k tanh(k h) por iteración de punto fijo."""
    omega = 2.0 * np.pi / T
    k = omega**2 / g                          # estimación de aguas profundas
    for _ in range(itmax):
        k_new = omega**2 / (g * np.tanh(k * h))
        if abs(k_new - k) < tol * max(1.0, k_new):
            return k_new
        k = k_new
    return k


# -----------------------------------------------------------------------------
# Envolventes con sus transformadas analíticas
#     hat_B(lambda) = ∫ B(y) exp(-i lambda y) dy
# -----------------------------------------------------------------------------

@dataclass
class Envelope:
    a: float
    b: float
    B:    Callable[[np.ndarray], np.ndarray]
    Bhat: Callable[[np.ndarray], np.ndarray]
    name: str = ""


def erf_rectangle(a: float, b: float, sigma: float) -> Envelope:
    """
    Rectangular suavizada por convolución con gaussiana de ancho sigma.
        B(y) = 0.5 * (erf((y-a)/(sigma*sqrt(2))) - erf((y-b)/(sigma*sqrt(2))))
        hat_B(lambda) = exp(-sigma^2 lambda^2 / 2)
                        * (exp(-i lambda a) - exp(-i lambda b)) / (i lambda)
        hat_B(0) = b - a
    """
    from scipy.special import erf
    s2 = sigma * np.sqrt(2.0)

    def B(y):
        y = np.asarray(y, dtype=float)
        return 0.5 * (erf((y - a) / s2) - erf((y - b) / s2))

    def Bhat(lam):
        lam = np.asarray(lam, dtype=float)
        out = np.empty(lam.shape, dtype=complex)
        small = np.abs(lam) < 1e-12
        out[small] = (b - a)
        L = lam[~small]
        out[~small] = (np.exp(-0.5 * sigma**2 * L**2)
                       * (np.exp(-1j * L * a) - np.exp(-1j * L * b))
                       / (1j * L))
        return out

    return Envelope(a, b, B, Bhat, name=f"erf(σ={sigma:g})")


def tukey_rectangle(a: float, b: float, alpha: float) -> Envelope:
    """
    Tukey window (cosine-tapered) con soporte exacto [a, b]. La fracción
    alpha se reparte mitad y mitad en los bordes (alpha=0 -> rect pura,
    alpha=1 -> Hann completa). hat_B se evalúa por cuadratura.
    """
    L = b - a

    def B(y):
        y = np.asarray(y, dtype=float)
        out = np.zeros(y.shape, dtype=float)
        inside = (y >= a) & (y <= b)
        t = (y[inside] - a) / L                # t en [0, 1]
        w = np.ones_like(t)
        if alpha > 0:
            left  = t < alpha / 2
            right = t > 1 - alpha / 2
            w[left]  = 0.5 * (1 + np.cos(np.pi * (2 * t[left] / alpha - 1)))
            w[right] = 0.5 * (1 + np.cos(np.pi * (2 * t[right] / alpha
                                                  - 2 / alpha + 1)))
        out[inside] = w
        return out

    from scipy.integrate import quad

    def Bhat_scalar(lam: float) -> complex:
        re, _ = quad(lambda y: B(np.array([y]))[0] * np.cos(lam * y),
                     a, b, limit=200)
        im, _ = quad(lambda y: -B(np.array([y]))[0] * np.sin(lam * y),
                     a, b, limit=200)
        return re + 1j * im

    def Bhat(lam):
        lam = np.atleast_1d(np.asarray(lam, dtype=float))
        return np.array([Bhat_scalar(float(L_)) for L_ in lam])

    return Envelope(a, b, B, Bhat, name=f"tukey(α={alpha:g})")


# -----------------------------------------------------------------------------
# Aplicación de la envolvente a una tabla de Celeris
# -----------------------------------------------------------------------------

def apply_envelope_to_table(
    table: np.ndarray,
    h: float,
    envelope: Envelope,
    y_max: float,
    pad: float = 6.0,
    samples_per_lobe: int = 8,
    amplitude_threshold: float = 1e-4,
) -> np.ndarray:
    """
    Parameters
    ----------
    table : (N, 4) array con columnas [|hat_eta|, T, theta_rad, phi_rad]
    h     : profundidad usada en la dispersión
    envelope : Envelope con (a, b, B, Bhat)
    y_max : máximo |y| (m) en el que la nueva tabla debe reproducir bien la
        señal; controla el espaciado en λ para evitar aliasing en la suma de
        Riemann (Δλ * y_max ≤ π / samples_per_lobe).
    pad : extensión de la grilla en λ más allá del rango de los λ_i originales,
        en unidades del ancho espectral característico de B (≈ 2π/(b-a)).
    samples_per_lobe : factor extra de sobre-muestreo en λ.
    amplitude_threshold : umbral relativo para descartar modos despreciables
        en la tabla resultante.

    Returns
    -------
    new_table : (M, 4) array con columnas [|hat_eta|', T, theta_rad', phi_rad'].
        Cada fila i del input genera una banda de modos con el mismo T
        (la envolvente no afecta omega) y una distribución de θ que muestrea
        la convolución hat_B(λ - λ_i).
    """
    table = np.asarray(table, dtype=float)
    A_in, T_in, th_in, phi_in = table.T

    # Λ y |κ| para cada fila original
    kappa_i  = np.array([wavenumber_from_period(T, h) for T in T_in])
    lambda_i = kappa_i * np.sin((th_in))

    # Grilla de lambdas: cubre los λ_i + ancho espectral de B + padding.
    L = envelope.b - envelope.a
    d_lambda_env = 2.0 * np.pi / L
    lam_min = lambda_i.min() - pad * d_lambda_env
    lam_max = lambda_i.max() + pad * d_lambda_env
    # espaciado mínimo dictado por y_max (criterio de Nyquist espacial)
    d_lam = (np.pi / y_max) / samples_per_lobe
    n_lam = int(np.ceil((lam_max - lam_min) / d_lam)) + 1
    lam_grid = np.linspace(lam_min, lam_max, n_lam)
    d_lam = lam_grid[1] - lam_grid[0]

    rows = []
    for i in range(len(T_in)):
        # Convolución en λ: para cada modo i, hat_B(λ - λ_i).
        Bhat_shift = envelope.Bhat(lam_grid - lambda_i[i])
        # Peso complejo del modo original (en formato coseno):
        c_i = A_in[i] * np.exp(1j * (phi_in[i]))
        # Densidad espectral resultante en λ (a omega = omega_i fija). El
        # factor de discretización Δλ/(2π) convierte la integral inversa
        # de Fourier en una suma de modos discretos.
        coef = (d_lam / (2.0 * np.pi)) * c_i * Bhat_shift   # complejo

        amp   = np.abs(coef)
        phase = np.angle(coef)

        # θ' a partir de λ' = |κ_i| sin(θ'). Sólo conservamos modos
        # propagantes |λ'| ≤ |κ_i| (los evanescentes no se generan).
        sin_th = lam_grid / kappa_i[i]
        prop = np.abs(sin_th) <= 1.0
        theta_p = np.arcsin(np.clip(sin_th[prop], -1.0, 1.0))

        for amp_j, theta_j, phi_j in zip(amp[prop], theta_p, phase[prop]):
            rows.append((amp_j, T_in[i], theta_j, phi_j))

    new_table = np.array(rows)

    if amplitude_threshold > 0 and len(new_table):
        amax = new_table[:, 0].max()
        new_table = new_table[new_table[:, 0] >= amplitude_threshold * amax]

    return new_table


# -----------------------------------------------------------------------------
# Reconstrucción directa de eta a partir de una tabla (para verificar)
# -----------------------------------------------------------------------------

def eta_from_table(table: np.ndarray, h: float,
                   y: np.ndarray, t: float) -> np.ndarray:
    """eta(y, t) = sum_i A_i cos(lambda_i y - omega_i t + phi_i)."""
    eta = np.zeros_like(y, dtype=float)
    for A, T, th, phi in table:
        kappa = wavenumber_from_period(T, h)
        lam   = kappa * np.sin((th))
        omega = 2.0 * np.pi / T
        eta  += A * np.cos(lam * y - omega * t + (phi))
    return eta
