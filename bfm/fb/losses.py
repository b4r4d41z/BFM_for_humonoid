"""Successor-measure TD estimator, matching Meta Motivo's basic FB update.

Rows are transitions i; columns are their next states j. Only i=j is the
immediate successor. Off-diagonal columns are independent replay references.
"""
import torch


def twin_reduce(predictions, pessimism):
    # Two-network version of mean - penalty * pairwise absolute disagreement.
    return predictions.mean(0) - pessimism * (predictions[0] - predictions[1]).abs()


def fb_loss(forwards, backwards, target_forwards, target_backwards, discount, *, ortho_coef=1., pessimism=0.):
    n = backwards.shape[0]
    if n < 2 or forwards.shape != (2, n, backwards.shape[-1]) or discount.shape != (n, 1):
        raise ValueError("FB requires twin [2,N,D] forwards, [N,D] backwards, [N,1] discounts and N>=2")
    off = ~torch.eye(n, dtype=torch.bool, device=backwards.device)
    with torch.no_grad():
        target = twin_reduce(target_forwards @ target_backwards.T, pessimism)
        bootstrap = discount * target
    matrices = forwards @ backwards.T
    residual = matrices - bootstrap.unsqueeze(0)
    off_diagonal = 0.5 * residual[:, off].square().sum(0).mean()
    # Subtracting detached target diagonals changes the scalar diagnostic only,
    # not the immediate-transition gradient -sum_k mean_i F_k(i)^T B(i).
    diagonal = -residual.diagonal(dim1=-2, dim2=-1).sum(0).mean()
    gram = backwards @ backwards.T
    orthogonality = 0.5 * gram[off].square().mean() - gram.diag().mean()
    td = off_diagonal + diagonal
    return td + ortho_coef * orthogonality, {
        "fb_td_loss": td, "fb_off_diagonal_loss": off_diagonal,
        "fb_diagonal_loss": diagonal, "backward_orthogonality_loss": orthogonality,
    }


def actor_loss(forwards, z):
    # mean - .5 * disagreement is exactly the min for a twin ensemble.
    return -twin_reduce((forwards * z.unsqueeze(0)).sum(-1), 0.5).mean()
