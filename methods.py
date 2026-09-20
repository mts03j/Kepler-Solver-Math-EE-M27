import numpy as np


def f(E, e, M):
    return E - e * np.sin(E) - M


def fp(E, e):
    return 1 - e * np.cos(E)


def bisection(e, M, tol=1e-8, max_iter=100):

    a = M
    b = M+e
    history = []

    iterations = 0

    while iterations < max_iter:

        c = (a + b) / 2
        history.append(c)

        if abs(f(c, e, M)) < tol:
            return c, iterations + 1, history

        if f(a, e, M) * f(c, e, M) < 0:
            b = c
        else:
            a = c

        iterations += 1

    return None, iterations, history


def regula_falsi(e, M, tol=1e-8, max_iter=100):

    a = M
    b = M+e
    history = []

    iterations = 0

    while iterations < max_iter:

        fa = f(a, e, M)
        fb = f(b, e, M)

        c = b - fb * (b - a) / (fb - fa)
        history.append(c)

        if abs(f(c, e, M)) < tol:
            return c, iterations + 1, history

        if f(a, e, M) * f(c, e, M) < 0:
            b = c
        else:
            a = c

        iterations += 1

    return None, iterations, history


def newton_raphson(e, M, tol=1e-8, max_iter=100):

    r = M + e / 2
    iterations = 0
    history = [r]

    while iterations < max_iter:
        fr = f(r, e, M)

        if abs(fr) < tol:
            return r, iterations, history

        fpr = fp(r, e)

        if abs(fpr) < 1e-14:
            return None, iterations, history

        r = r - fr / fpr
        history.append(r)
        iterations += 1

    return None, iterations, history


def markley_kepler(e, M):
    """
    Solve M = E - e*sin(E) using Markley's fifth-order method.

    Assumes an elliptical orbit with 0 <= e < 1.
    Returns the solution for the mean anomaly reduced to [-pi, pi).
    """
    if not 0 <= e < 1:
        raise ValueError(
            "Eccentricity must satisfy 0 <= e < 1 for elliptical orbits."
        )

    # Reduce the mean anomaly to [-pi, pi)
    M = np.mod(M + np.pi, 2 * np.pi) - np.pi

    # For M = 0, the exact solution is E = 0
    if M == 0.0:
        return 0.0

    alpha = (
        3 * np.pi**2
        + 1.6 * np.pi * (np.pi - abs(M)) / (1 + e)
    ) / (np.pi**2 - 6)

    d = 3 * (1 - e) + alpha * e
    q = 2 * alpha * d * (1 - e) - M**2
    r = 3 * alpha * d * (d - 1 + e) * M + M**3

    w = (abs(r) + np.sqrt(q**3 + r**2)) ** (2 / 3)

    E1 = (
        (2 * r * w) / (w**2 + w * q + q**2) + M
    ) / d

    if e > 0.5 and abs(E1) < 1.0:
        numerator = (
            -1.7454287843856404e-6 * E1**6
            + 4.1584640418181644e-4 * E1**4
            - 3.0956446448551138e-2 * E1**2
            + 1
        )

        denominator = (
            1.7804367119519884e-8 * E1**8
            + 5.9727613731070647e-6 * E1**6
            + 1.0652873476684142e-3 * E1**4
            + 1.1426132130869317e-1 * E1**2
            + 6
        )

        M_star = (1 - e) * E1 + e * E1**3 * (
            numerator / denominator
        )
    else:
        M_star = E1 - e * np.sin(E1)

    f_E = M_star - M
    fp_E = 1 - e + 2 * e * np.sin(E1 / 2) ** 2
    fpp_E = E1 - M_star
    fppp_E = 1 - fp_E
    fpppp_E = -fpp_E

    d3 = -f_E / (
        fp_E - 0.5 * f_E * fpp_E / fp_E
    )

    d4 = -f_E / (
        fp_E
        + 0.5 * d3 * fpp_E
        + (1 / 6) * d3**2 * fppp_E
    )

    d5 = -f_E / (
        fp_E
        + 0.5 * d4 * fpp_E
        + (1 / 6) * d4**2 * fppp_E
        + (1 / 24) * d4**3 * fpppp_E
    )

    return E1 + d5
