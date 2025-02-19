import torch
import torch.nn.functional as F
from entmax.activations import sparsemax, entmax15
from entmax.root_finding import normmax_bisect, entmax_bisect


def BaseLoss(theta, y, omega_func, activation_func, dim=-1):
    y_pred = activation_func(theta, dim=dim)
    omega_conj = torch.sum(theta * y_pred, dim=-1) - omega_func(y_pred)
    loss = omega_conj + omega_func(y) - torch.sum(y * theta, dim=-1)
    return loss #loss.mean()


def omega_sparsemax(x):
    return 0.5 * torch.sum(x**2, dim=1)  # - 0.5


def omega_softmax(x):
    eps = 1e-7
    x = torch.clamp(x, eps, 1.0 - eps)
    entropy = -torch.sum(x * torch.log(x), dim=-1)
    return -entropy


def omega_normmax(x, alpha=5):
    return -(1 - torch.norm(x, p=alpha, dim=-1))


def omega_entmax(x, alpha=1.5):
    entropy = (1 - (x**alpha).sum(dim=-1)) / (alpha * (alpha - 1))
    return -entropy


def softmax_loss(theta, y):
    return BaseLoss(theta, y, omega_softmax, F.softmax)


def sparsemax_loss(theta, y):
    return BaseLoss(theta, y, omega_sparsemax, sparsemax)


def normmax_loss(theta, y, alpha=5, n_iter=50):
    def _omega_normax(x):
        return omega_normmax(x, alpha=alpha)

    def _normmax(x, dim):
        return normmax_bisect(x, alpha=alpha, dim=dim, n_iter=n_iter)

    return BaseLoss(theta, y, _omega_normax, _normmax)


def entmax15_loss(theta, y):
    def _omega_entmax15(x):
        return omega_entmax(x, alpha=1.5)

    def _entmax15(x, dim):
        return entmax15(x, dim=0)

    return BaseLoss(theta, y, _omega_entmax15, _entmax15)


def entmax_loss(theta, y, alpha=1.5, n_iter=50):
    def _omega_entmax(x):
        return omega_entmax(x, alpha=alpha)

    def _entmax(x, dim):
        return entmax_bisect(x, alpha=alpha, dim=-1, n_iter=n_iter)

    return BaseLoss(theta, y, _omega_entmax, _entmax)
