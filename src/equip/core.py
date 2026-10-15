import numpy as np


def simplex(x):
    u = np.sort(x)[::-1]
    css = np.cumsum(u) - 1
    rho = np.flatnonzero(u - css / np.arange(1, len(u) + 1) > 0)[-1]
    return np.maximum(x - css[rho] / (rho + 1), 0)


def aggregate(messages, weights, style, mask=None):
    x = np.asarray(messages, float)
    scales = np.asarray(weights) * np.asarray(style)
    if mask is not None:
        scales = scales * np.asarray(mask)
    logits = (x * scales[:, None]).sum(0)
    logits -= logits.max()
    result = np.exp(logits)
    return result / result.sum()


def influence(messages, weights, style, choice):
    n = len(weights)
    full = aggregate(messages, weights, style)[choice]
    values = []
    for i in range(n):
        mask = np.ones(n)
        mask[i] = 0
        values.append(full - aggregate(messages, weights, style, mask)[choice])
    return np.array(values)


def control(messages, weights, style, choice, quality, upper_debt, lower_debt, step):
    def objective(w):
        utility = float(w @ quality)
        effect = influence(messages, w, style, choice)
        return utility - float((upper_debt - lower_debt) @ effect)

    gradient = np.zeros(len(weights))
    for i in range(len(weights)):
        perturb = np.zeros(len(weights))
        perturb[i] = 1e-4
        gradient[i] = (
            objective(simplex(weights + perturb))
            - objective(simplex(weights - perturb))
        ) / 2e-4
    return simplex(weights + step * gradient)
