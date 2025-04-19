import torch


def centering(x: torch.Tensor) -> torch.Tensor:
    n = x.shape[0]
    unit_matrix = torch.ones([n, n])
    identity = torch.eye(unit_matrix)
    centering_matrix = identity - 1 / n * unit_matrix

    return centering_matrix


def linear_hsic(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    l_x = x @ x.T
    l_y = y @ y.T

    return torch.sum(centering(l_x) * centering(l_y))


def linear_cka(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    hsic = linear_hsic(x, y)
    var_x = linear_hsic(x, x)
    var_y = linear_hsic(y, y)

    return hsic / torch.sqrt(var_x * var_y)
