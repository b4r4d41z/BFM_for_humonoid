

from ` /.../BFM_for_humonoid/Test_training/source/Test_training `

```bash
python -m pip install -e .
```

start env with agent from ` /.../BFM_for_humonoid/Test_training `

```bash 
python scripts/zero_agent.py --task Template-Test-Training-Direct-v0
```

: it creates the Gym environment for the given task ID and steps it with zero (or fixed) actions to quickly verify that the scene/environment initializes correctly and the simulation starts.