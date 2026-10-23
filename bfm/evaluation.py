"""Measured closed-loop evaluation, usable with Isaac or an explicitly labelled mock."""
import torch


class TaskMetrics:
    def __init__(self, task, *, reach_distance=0.06, target_half_extent=(0.06, 0.06, 0.04), hold_steps=30, dt=1/60):
        if task not in ("reach_red_ball", "move_red_ball_to_container"):
            raise ValueError("Unknown evaluation task")
        if reach_distance <= 0 or hold_steps < 1 or dt <= 0 or any(x <= 0 for x in target_half_extent):
            raise ValueError("Invalid evaluation thresholds")
        self.task, self.threshold, self.extent = task, reach_distance, torch.tensor(target_half_extent)
        self.hold_steps, self.dt = hold_steps, dt
        self.steps, self.held, self.first_success = 0, 0, None
        self.initial, self.final, self.minimum = None, None, float("inf")
        self.initial_ball_distance, self.ball_distance = None, None

    def observe(self, state, *, initial=False):
        def vec(key):
            x = torch.as_tensor(state[key]).detach().cpu()
            if x.shape != (1, 3) or not torch.isfinite(x).all():
                raise ValueError("Evaluation requires finite single-environment measured positions")
            return x[0]
        eef, ball, target = [vec(k) for k in ("right_eef_position", "red_ball_position", "container_position")]
        distance = float((eef-ball).norm())
        self.ball_distance = float((ball-target).norm())
        if initial:
            self.initial, self.initial_ball_distance = distance, self.ball_distance
        else:
            self.steps += 1
        self.final, self.minimum = distance, min(self.minimum, distance)
        if initial:
            return
        inside = bool(((ball-target).abs() <= self.extent).all())
        self.held = self.held + 1 if inside else 0
        success = distance <= self.threshold if self.task == "reach_red_ball" else self.held >= self.hold_steps
        if success and self.first_success is None:
            self.first_success = self.steps * self.dt

    def result(self):
        return {"success": self.first_success is not None,
                "initial_eef_to_ball_distance": self.initial, "final_eef_to_ball_distance": self.final,
                "minimum_eef_to_ball_distance": self.minimum, "time_to_success": self.first_success,
                "episode_length": self.steps, "episode_duration": self.steps * self.dt,
                "initial_ball_to_container_distance": self.initial_ball_distance,
                "final_ball_to_container_distance": self.ball_distance,
                "target_hold_steps_at_end": self.held}


def evaluate_episode(env, bridge, runner, *, seed, task, max_steps, **metric_options):
    if max_steps < 1:
        raise ValueError("Evaluation requires at least one control step")
    env.reset(seed=seed)
    runner.reset()
    state = bridge.extract()
    metrics = TaskMetrics(task, dt=runner.contract.control_dt, **metric_options)
    metrics.observe(state, initial=True)
    terminated, truncated = False, False
    for _ in range(max_steps):
        action = runner.act(state)
        state, term, trunc, _, _ = bridge.step(action)
        metrics.observe(state)
        terminated, truncated = bool(term[0]), bool(trunc[0])
        if terminated or truncated:
            break
    return {"seed": seed, "task": task, "terminated": terminated, "truncated": truncated,
            "evaluation_cutoff": not (terminated or truncated), **metrics.result()}
