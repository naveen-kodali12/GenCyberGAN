import torch


def gradient_penalty_flow(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, 1, device=device)
    interp = (alpha * real + (1-alpha) * fake).requires_grad_(True)
    score = critic(interp)
    grad = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.reshape(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def gradient_penalty_graph(critic, real, fake, device):
    alpha = torch.rand(real.size(0), 1, device=device)
    interp = (alpha * real + (1-alpha) * fake).requires_grad_(True)
    score = critic(interp)
    grad = torch.autograd.grad(score, interp, torch.ones_like(score), create_graph=True, retain_graph=True, only_inputs=True)[0]
    return ((grad.reshape(grad.size(0), -1).norm(2, dim=1) - 1) ** 2).mean()


def temporal_smoothness_loss(x):
    if x.size(1) < 2:
        return torch.tensor(0., device=x.device)
    return (x[:,1:,:] - x[:,:-1,:]).abs().mean()


def graph_flow_coherence_loss(x, g):
    activity = x.abs().mean(dim=(1,2))
    graph_activity = g[:, [0,1,2,6]].mean(dim=1)
    activity = (activity - activity.mean()) / (activity.std() + 1e-6)
    graph_activity = (graph_activity - graph_activity.mean()) / (graph_activity.std() + 1e-6)
    return ((activity - graph_activity) ** 2).mean()
