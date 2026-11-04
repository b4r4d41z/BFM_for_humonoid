import copy
import math
import pytest
import torch

from bfm.fb.agent import FBAgent, TrainConfig
from bfm.fb.model import FBModel, ModelConfig
from bfm.fb.losses import fb_loss, actor_loss
from bfm.data.dataset import BFMDataset
from bfm.data.replay_buffer import BFMReplayBuffer
from bfm.fb.checkpoint import save_checkpoint, load_model, load_payload, resume_agent


def agent():
    return FBAgent(ModelConfig(hidden_dim=32, latent_dim=4), TrainConfig(tau=.1))


def test_objective_matches_explicit_reference_terms():
    torch.manual_seed(2)
    f = torch.randn(2, 5, 3, requires_grad=True)
    b = torch.randn(5, 3, requires_grad=True)
    tf = torch.randn_like(f, requires_grad=True)
    tb = torch.randn_like(b, requires_grad=True)
    d = torch.tensor([[0.], [.9], [.9], [.9], [0.]])
    total, metrics = fb_loss(f, b, tf, tb, d, ortho_coef=.3)
    off, diag, orth = 0., 0., 0.
    for i in range(5):
        diag += -sum(f[k, i] @ b[i] - d[i, 0] * (tf[:, i].detach() @ tb[i].detach()).mean() for k in range(2)) / 5
        orth -= b[i].square().sum() / 5
        for j in range(5):
            if i == j: continue
            for k in range(2):
                residual = f[k, i] @ b[j] - d[i, 0] * (tf[:, i].detach() @ tb[j].detach()).mean()
                off += .5 * residual.square() / 20
            orth += .5 * (b[i] @ b[j]).square() / 20
    assert torch.allclose(total, off + diag + .3 * orth)
    total.backward()
    assert f.grad.abs().sum() > 0 and b.grad.abs().sum() > 0
    assert tf.grad is None and tb.grad is None
    assert torch.allclose(metrics["backward_orthogonality_loss"], orth)


def test_tabular_terminal_immediate_successor_correction():
    # Uniform rho over 3 terminal successors: density = 3 * I. A complete
    # batch has zero off-diagonal TD residual and only the Dirac gradient.
    b = math.sqrt(3) * torch.eye(3)
    f = b.repeat(2, 1, 1).requires_grad_()
    loss, metrics = fb_loss(f, b, torch.zeros_like(f), b, torch.zeros(3, 1), ortho_coef=0)
    assert metrics["fb_off_diagonal_loss"] == 0
    assert metrics["fb_diagonal_loss"] == -6
    loss.backward()
    assert torch.allclose(f.grad, -b.repeat(2, 1, 1) / 3)


def test_known_tabular_successor_is_expected_td_fixed_point():
    # Enumerate every iid two-transition batch in a 3-state deterministic cycle.
    # rho is uniform and B=sqrt(3)*I, so F=sqrt(3)*(I-gamma*P)^-1 P.
    # The expected semi-gradient must vanish at the known successor density.
    p = torch.tensor([[0., 1., 0.], [0., 0., 1.], [1., 0., 0.]], dtype=torch.float64)
    gamma = .8
    solution = math.sqrt(3) * torch.linalg.solve(torch.eye(3, dtype=torch.float64) - gamma*p, p)
    forward_table = solution.repeat(2, 1, 1).requires_grad_()
    b = math.sqrt(3) * torch.eye(3, dtype=torch.float64)
    total = 0.
    for i in range(3):
        for j in range(3):
            rows = torch.tensor([i, j])
            successors = (rows + 1) % 3
            loss, _ = fb_loss(forward_table[:, rows], b[successors],
                              forward_table[:, successors].detach(), b[successors],
                              torch.full((2, 1), gamma, dtype=torch.float64), ortho_coef=0)
            total = total + loss / 9
    total.backward()
    assert forward_table.grad.abs().max() < 1e-12


def test_shapes_latents_goal_reward_and_conditioning():
    m = FBModel(ModelConfig(state_dim=6, action_dim=2, hidden_dim=16, latent_dim=4))
    s = torch.randn(8, 6)
    z = m.sample_z(8)
    a = m.act(s, z)
    assert a.shape == (8, 2) and a.abs().max() <= 1
    assert m.forward_map(s, a, z).shape == (2, 8, 4)
    assert torch.allclose(z.norm(dim=-1), torch.full((8,), 2.))
    assert not torch.allclose(a, m.act(s, -z))
    assert m.goal_inference(s).shape == z.shape
    reward_z = m.reward_inference(s, torch.arange(8.))
    assert reward_z.norm().item() == pytest.approx(2.)
    with pytest.raises(ValueError): m.reward_inference(s, torch.zeros(8))


def test_actor_uses_latent_value_and_freezes_representations():
    a = agent()
    s, z = torch.randn(5, 28), a.model.sample_z(5)
    before = [p.clone() for p in a.fb_parameters]
    actor_before = [p.clone() for p in a.model.actor.parameters()]
    a.update_actor(s, z)
    assert all(torch.equal(p, old) and p.grad is None for p, old in zip(a.fb_parameters, before))
    assert any(not torch.equal(p, old) for p, old in zip(a.model.actor.parameters(), actor_before))
    f = torch.tensor([[[1., 2.]], [[3., 4.]]])
    assert actor_loss(f, torch.tensor([[1., 0.]])) == -1
    assert actor_loss(f, torch.tensor([[0., 1.]])) == -2


def test_updates_targets_and_deterministic_resume(manifest, tmp_path):
    torch.manual_seed(42)
    data = BFMDataset(manifest)
    replay = BFMReplayBuffer(data.episodes, seed=3)
    a = agent()
    a.model.normalizer.fit(replay.data["observation"])
    old = copy.deepcopy(a.model.state_dict())
    target_before = [p.clone() for p in a.target_F.parameters()]
    for _ in range(10):
        metrics = a.update(replay.sample_transitions(8))
        assert all(math.isfinite(v) for v in metrics.values())
    assert any(not torch.equal(old[k], v) for k, v in a.model.state_dict().items())
    assert all(not p.requires_grad and p.grad is None for p in a.target_F.parameters())
    target_before = [p.clone() for p in a.target_F.parameters()]
    a.soft_update()
    for old_t, target, source in zip(target_before, a.target_F.parameters(), a.model.forwards.parameters()):
        assert torch.allclose(target, .9 * old_t + .1 * source)
    ref, idx = replay.reference_states(10, 1)
    checkpoint = tmp_path / "fb.pt"
    save_checkpoint(checkpoint, a, replay, data, ref, idx, {})
    m, payload = load_model(checkpoint)
    z = a.model.sample_z(3)
    s = ref[:3]
    assert torch.equal(m.act(s, z), a.model.act(s, z))
    # Restore RNG to checkpoint before both continuation paths.
    resume_agent(payload, a, replay, data)
    expected = a.update(replay.sample_transitions(8))
    b, rb = agent(), BFMReplayBuffer(data.episodes)
    resume_agent(load_payload(checkpoint), b, rb, data)
    actual = b.update(rb.sample_transitions(8))
    assert actual == expected
    assert all(torch.equal(a.model.state_dict()[k], v) for k, v in b.model.state_dict().items())


def test_termination_masks_bootstrap_but_time_limit_does_not(manifest, monkeypatch):
    import bfm.fb.agent as module
    data = BFMDataset(manifest)
    replay = BFMReplayBuffer(data.episodes)
    batch = replay.sample_transitions(4)
    batch["terminated"] = torch.tensor([True, False, False, False])
    batch["truncated"] = torch.tensor([False, True, False, False])
    batch["duration"] = torch.tensor([.1, .05, .1, .1])
    real_loss = module.fb_loss
    observed = {}
    def capture(f, b, tf, tb, discount, **kwargs):
        observed["discount"] = discount.detach().clone()
        return real_loss(f, b, tf, tb, discount, **kwargs)
    monkeypatch.setattr(module, "fb_loss", capture)
    a = agent()
    a.update(batch)
    assert torch.allclose(observed["discount"].ravel(), torch.tensor([0., math.sqrt(.99), .99, .99]))


def test_orthogonality_changes_backward_gradient():
    torch.manual_seed(8)
    f, b = torch.randn(2, 4, 3), torch.randn(4, 3, requires_grad=True)
    target = torch.zeros_like(f)
    base, _ = fb_loss(f, b, target, b.detach(), torch.zeros(4, 1), ortho_coef=0)
    regularized, _ = fb_loss(f, b, target, b.detach(), torch.zeros(4, 1), ortho_coef=1)
    assert not torch.allclose(torch.autograd.grad(base, b)[0], torch.autograd.grad(regularized, b)[0])
