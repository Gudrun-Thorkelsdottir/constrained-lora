import torch



@torch.no_grad()
def estimate_path_length(u, x, n_steps):
    """
    Estimates average path length over a vector field trajectory.

    Args:
        u: Callable, the vector field function u_theta(x, t)
        x: Tensor of shape [batch_size, dim], initial samples
        n_steps: Integer, number of Euler steps

    Returns:
        L: Scalar tensor, average path length across batch
    """


    delta_t = 1.0 / n_steps
    t = 0.0
    length = 0.0


    for _ in range(n_steps):

        v = u(x, t)
        x_next = x + delta_t * v

        length += torch.norm(x_next - x, dim=1).sum()

        x = x_next
        t += delta_t


    L = length / x.shape[0]
    return L
