

from ` /.../BFM_for_humonoid/Test_training/source/Test_training `

```bash
python -m pip install -e .
```

start env with agent from ` /.../BFM_for_humonoid/Test_training `

```bash 
python scripts/zero_agent.py --task Template-Test-Training-Direct-v0 --max_steps 5000 --log_every 200 --watchdog_sec 10
```

It creates the Gym environment for the given task ID and steps it with zero (or fixed) actions to quickly verify that the scene/environment initializes correctly and the simulation starts.

---

```bash 
python scripts/rsl_rl/train.py --task Template-Test-Training-Direct-v0 --num_envs 1 --seed 0
```


 