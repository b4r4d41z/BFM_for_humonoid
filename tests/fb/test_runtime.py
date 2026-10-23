from types import SimpleNamespace as NS
import subprocess
import sys

import pytest
import torch

from bfm.contracts.kuavo import ARM_JOINTS, RIGHT_FINGERS, BFMActionAdapter
from bfm.data.dataset import BFMDataset
from bfm.data.replay_buffer import BFMReplayBuffer
from bfm.data.recorder import BFMRecorder, cpu_state
from bfm.data.zarr_io import read_episode
from bfm.fb.agent import FBAgent, TrainConfig
from bfm.fb.model import ModelConfig
from bfm.fb.checkpoint import save_checkpoint
from bfm.evaluation import evaluate_episode, TaskMetrics
from isaaclab_wrapper.bfm_state import IsaacTransitionBridge, contract_from_env
from isaaclab_wrapper.bfm_policy_runner import BFMPolicyRunner


class MockIsaac:
    """Tests interface ordering and auto-reset timing; not a physics simulator."""
    def __init__(self, horizon=15):
        self.unwrapped, self.device, self.num_envs = self, "cpu", 1
        names = list(reversed(ARM_JOINTS + RIGHT_FINGERS))
        self.robot = NS(joint_names=names, body_names=["zarm_r7_link"], data=NS(
            joint_pos=torch.zeros(1, len(names)), joint_vel=torch.zeros(1, len(names)),
            default_joint_pos=torch.zeros(1, len(names)),
            soft_joint_pos_limits=torch.tensor([[[-1., 1.]] * len(names)]),
            body_pos_w=torch.zeros(1, 1, 3), body_quat_w=torch.tensor([[[1., 0., 0., 0.]]]),
        ))
        self._ctrl_joint_names = list(ARM_JOINTS)
        self._joint_pos_target = torch.zeros(1, 14)
        self._left_claw_closed, self._right_claw_closed = torch.zeros(1, dtype=torch.bool), torch.zeros(1, dtype=torch.bool)
        self.scene = NS(env_origins=torch.tensor([[4., 0., 0.]]))
        self.red_ball = NS(data=NS(root_pos_w=torch.tensor([[4.4, -.1, .2]]), root_lin_vel_w=torch.zeros(1, 3)))
        self.container_base = NS(data=NS(root_pos_w=torch.tensor([[4.5, .1, .2]])))
        self.cfg = NS(sim=NS(dt=1/120), decimation=2, ball_radius=.035, container_base_cfg=NS(spawn=NS(size=(.22, .22, .025))))
        self.horizon, self.tick = horizon, 0
        self.reset()

    def reset(self, seed=None):
        self.tick = 0
        self.robot.data.joint_pos.zero_()
        self.robot.data.joint_vel.zero_()
        self.robot.data.body_pos_w[:] = self.scene.env_origins[:, None]
        self._left_claw_closed.zero_(); self._right_claw_closed.zero_()
        return {}, {}

    def step(self, action):
        ids = [self.robot.joint_names.index(n) for n in ARM_JOINTS]
        self._joint_pos_target = action[:, :14].clamp(-1, 1).clone()
        old = self.robot.data.joint_pos[:, ids].clone()
        new = old + .1 * (self._joint_pos_target - old)
        self.robot.data.joint_pos[:, ids] = new
        self.robot.data.joint_vel[:, ids] = (new-old) * 60
        self.robot.data.body_pos_w[:, 0] = new[:, :3] + self.scene.env_origins
        self._left_claw_closed = action[:, 14] > 30
        self._right_claw_closed = action[:, 20] > 30
        self.tick += 1
        term, trunc = torch.tensor([False]), torch.tensor([self.tick >= self.horizon])
        if getattr(self, "_bfm_pre_reset_observer", None):
            self._bfm_pre_reset_observer(term, trunc)
        if trunc[0]: self.reset()
        return {}, torch.zeros(1), term, trunc, {}


@pytest.fixture
def checkpoint(manifest, tmp_path):
    torch.manual_seed(5)
    data = BFMDataset(manifest)
    replay = BFMReplayBuffer(data.episodes)
    agent = FBAgent(ModelConfig(hidden_dim=32, latent_dim=4), TrainConfig())
    agent.model.normalizer.fit(replay.data["observation"])
    for _ in range(3): agent.update(replay.sample_transitions(8))
    ref, idx = replay.reference_states(16, 1)
    path = tmp_path / "runtime.pt"
    save_checkpoint(path, agent, replay, data, ref, idx, {})
    return path


def test_record_measured_terminal_state_before_autoreset(tmp_path):
    env = MockIsaac()
    c = contract_from_env(env, eef_body="zarm_r7_link", eef_offset=[0., 0., 0.])
    bridge = IsaacTransitionBridge(env, c)
    state = bridge.extract()
    assert torch.allclose(state["red_ball_position"], torch.tensor([[.4, -.1, .2]]), atol=1e-6)
    recorder = BFMRecorder(c, episode_id="mock0", split="train", provenance={"kind": "mock_test"},
                           initial_scene={k: v.tolist() for k, v in cpu_state(state).items()}, joint_names=env.robot.joint_names)
    action = BFMActionAdapter(c)(torch.ones(1, 8) * .5)
    for _ in range(3):
        for substep in range(6):
            ns, term, trunc, executed, _ = bridge.step(action)
            if trunc[0]: break
        recorder.append(state, *executed, ns, terminated=term[0], truncated=trunc[0], control_steps=substep+1)
        state = ns
    assert env.tick == 0  # auto-reset happened
    assert state["right_arm_q"].abs().sum() > 0  # retained pre-reset observation
    assert executed[0][0, 7] == 1  # retained closed claw command, despite reset
    recorder.save(tmp_path / "episode.zarr")
    arrays, metadata = read_episode(tmp_path / "episode.zarr")
    assert arrays["next_step_index"].tolist() == [6, 12, 15]
    assert arrays["next_obs/right_arm_q"][-1].sum() > 0
    assert "all_joint_q" in metadata["state_fields"]
    bridge.close()


def test_runtime_checkpoint_goal_cache_reset_and_mock_rollout(checkpoint):
    env = MockIsaac()
    runner = BFMPolicyRunner(checkpoint)
    bridge = IsaacTransitionBridge(env, runner.contract)
    runner.set_goal({"type": "reward", "task": "reach_red_ball"})
    first_z = runner.z.clone()
    initial = bridge.extract()
    a = runner.act(initial)
    for _ in range(5): assert torch.equal(a, runner.act(initial))
    assert runner.scheduler.stats.inference_calls == 1
    runner.act(initial)
    assert runner.scheduler.stats.inference_calls == 2
    runner.reset()
    assert torch.equal(a, runner.act(initial))
    result = evaluate_episode(env, bridge, runner, seed=7, task="reach_red_ball", max_steps=30)
    assert result["episode_length"] == 15 and result["truncated"]
    assert result["final_eef_to_ball_distance"] >= 0
    runner.set_goal({"type": "reward", "task": "move_red_ball_to_container"})
    assert torch.isfinite(runner.z).all() and not torch.allclose(first_z, runner.z)
    runner.set_goal({"type": "goal", "state": initial})
    assert runner.act(initial).shape == (1, 26)
    with pytest.raises(ValueError): runner.set_goal({"type": "goal", "state": {"right_arm_q": [0.] * 7}})
    assert not any(p.requires_grad for p in runner.model.parameters())
    bridge.close()


def test_measured_container_dwell_success():
    m = TaskMetrics("move_red_ball_to_container", hold_steps=3)
    state = {k: torch.zeros(1, 3) for k in ("right_eef_position", "red_ball_position", "container_position")}
    m.observe(state, initial=True)
    m.observe(state); m.observe(state)
    assert not m.result()["success"]
    m.observe(state)
    assert m.result()["success"] and m.result()["time_to_success"] == pytest.approx(3/60)


def test_runtime_imports_no_bc_policy():
    subprocess.run([sys.executable, "-c", "import sys; import isaaclab_wrapper.bfm_policy_runner; assert not any(n.startswith('bc.') for n in sys.modules)"], check=True)
