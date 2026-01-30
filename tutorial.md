

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


 --- 

 5.234s] app ready
[7.109s] Simulation App Startup Complete
[INFO]: Parsing configuration from: Test_training.tasks.direct.test_training.test_training_env_cfg:TestTrainingEnvCfg
[INFO]: Parsing configuration from: Test_training.tasks.direct.test_training.agents.rsl_rl_ppo_cfg:PPORunnerCfg
[INFO] Logging experiment in directory: /home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/logs/rsl_rl/cartpole_direct
Exact experiment name requested from command line: 2026-01-30_18-20-43
Setting seed: 0
[7.767s] [ext: omni.physx.fabric-107.3.26] startup
[2026-01-30 18:20:43,943][ogn_registration][INFO] - Looking for Python nodes to register in omni.physx.fabric-107.3.26
[2026-01-30 18:20:43,943][ogn_registration][INFO] -  -> Registered nodes from module omni.physxfabric at /home/lab/.conda/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric-107.3.26+107.3.3.lx64.r.cp311.u353
[2026-01-30 18:20:43,944][ogn_registration][INFO] - Registering nodes in /home/lab/.conda/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric-107.3.26+107.3.3.lx64.r.cp311.u353 imported as omni.physxfabric with AutoNode config {}
[2026-01-30 18:20:43,944][ogn_registration][INFO] - Registering Python Node Types from omni.physxfabric at /home/lab/.conda/envs/env_isaaclab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric-107.3.26+107.3.3.lx64.r.cp311.u353 in omni.physx.fabric
[2026-01-30 18:20:43,944][ogn_registration][INFO] - ========================================================================================================================
[2026-01-30 18:20:43,944][ogn_registration][INFO] - No dependency on omni.graph, therefore no nodes to register in omni.physx.fabric
[2026-01-30 18:20:43,944][ogn_registration][INFO] - ...None found, no registration to do
[2026-01-30 18:20:43,944][ogn_registration][INFO] - ...Skipping: No OmniGraph presence in the module omni.physxfabric - No nodes in this module, do not remember it
[2026-01-30 18:20:43,944][ogn_registration][INFO] - Destroying registration record for omni.physx.fabric
[2026-01-30 18:20:43,944][ogn_registration][INFO] - OGN register omni.physx.fabric-107.3.26 took 822455.000000
[INFO]: Base environment:
        Environment device    : cuda:0
        Environment seed      : 0
        Physics step-size     : 0.008333333333333333
        Rendering step-size   : 0.016666666666666666
        Environment step-size : 0.016666666666666666
2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/l_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/l_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/l_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/lr_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/lr_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/lr_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/lr_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/lr_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/lr_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/rl_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rl_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rl_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/rl_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rl_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rl_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/r_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/r_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/r_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/ll_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/ll_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/ll_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/ll_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/ll_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/ll_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/rr_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rr_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rr_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/r_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/r_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/r_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/l_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/l_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/l_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/rr_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rr_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rr_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/zarm_l7_link/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/zarm_l7_link> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/zarm_l7_link/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,834ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_0/Robot/zarm_r7_link/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/zarm_r7_link> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/zarm_r7_link/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/l_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/l_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/l_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/l_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/l_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/l_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/ll_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/ll_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/ll_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/ll_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/ll_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/ll_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/lr_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/lr_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/lr_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/lr_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/lr_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/lr_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/zarm_r7_link/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/zarm_r7_link> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/zarm_r7_link/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/r_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/r_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/r_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/r_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/r_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/r_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/rl_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rl_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rl_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/rl_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rl_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rl_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/rr_foot_heel/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rr_foot_heel> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rr_foot_heel/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/rr_foot_toe/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/rr_foot_toe> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/rr_foot_toe/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

2026-01-30T10:20:44Z [7,845ms] [Warning] [omni.usd] Warning: in _ReportErrors at line 3172 of /builds/omniverse/usd-ci/USD/pxr/usd/usd/stage.cpp -- In </World/envs/env_1/Robot/zarm_l7_link/visuals>: Unresolved reference prim path @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_physics.usd@</visuals/zarm_l7_link> introduced by @/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/assets/robots/urdf/supported_biped_s40/configuration/supported_biped_s40_base.usd@</my_robot_description/zarm_l7_link/visuals> (recomposing stage on stage @anon:0x1215b3f0:World0.usd@ <0x1215c950>)

[INFO]: Time taken for scene creation : 0.112961 seconds
[INFO]: Scene manager:  <class InteractiveScene>
        Number of environments: 2
        Environment spacing   : 4.0
        Source prim name      : /World/envs/env_0
        Global prim paths     : []
        Replicate physics     : True
[INFO]: Starting the simulation. This may take a few seconds. Please wait...
2026-01-30T10:20:44Z [8,207ms] [Warning] [omni.fabric.plugin] getAttributeCount called on non-existent path /World/envs/env_1/Robot/rr_foot_toe/collisions/mesh_0
2026-01-30T10:20:44Z [8,207ms] [Warning] [omni.fabric.plugin] getTypes called on non-existent path /World/envs/env_1/Robot/rr_foot_toe/collisions/mesh_0
[INFO]: Time taken for simulation start : 0.835304 seconds
Creating window for environment.
ManagerLiveVisualizer cannot be created for manager: action_manager, Manager does not exist
ManagerLiveVisualizer cannot be created for manager: observation_manager, Manager does not exist
[INFO]: Completed setting up the environment...
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [25,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [26,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [27,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [28,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [29,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [30,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [31,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [96,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [97,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [98,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [99,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [100,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [101,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [102,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [103,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [104,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [105,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [106,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [107,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [108,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [109,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [110,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [111,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [112,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [113,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [114,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [115,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [116,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [117,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [118,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [119,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [120,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [121,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [122,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [123,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [124,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [125,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [126,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [127,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [64,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [65,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [66,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [67,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [68,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [69,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [70,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [71,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [72,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [73,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [74,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [75,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [76,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [77,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [78,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [79,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [80,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [81,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [82,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [83,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [84,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [85,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [86,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [87,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [88,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [89,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [90,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [91,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [92,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [93,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [94,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [95,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [32,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [33,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [34,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [35,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [36,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [37,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [38,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [39,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [40,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [41,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [42,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [43,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [44,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [45,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [46,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [47,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [48,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [49,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [50,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [51,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [52,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [53,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [54,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [55,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [56,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [57,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [58,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [59,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [60,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [61,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [62,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
/pytorch/aten/src/ATen/native/cuda/IndexKernel.cu:93: operator(): block: [0,0,0], thread: [63,0,0] Assertion `-sizes[i] <= index && index < sizes[i] && "index out of bounds"` failed.
Error executing job with overrides: []
Traceback (most recent call last):
  File "/home/lab/syp/IsaacLab/source/isaaclab_tasks/isaaclab_tasks/utils/hydra.py", line 101, in hydra_main
    func(env_cfg, agent_cfg, *args, **kwargs)
  File "/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/scripts/rsl_rl/train.py", line 164, in main
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/syp/IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/vecenv_wrapper.py", line 85, in __init__
    self.env.reset()
  File "/home/lab/.conda/envs/env_isaaclab/lib/python3.11/site-packages/gymnasium/wrappers/common.py", line 400, in reset
    return super().reset(seed=seed, options=options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/.conda/envs/env_isaaclab/lib/python3.11/site-packages/gymnasium/core.py", line 333, in reset
    return self.env.reset(seed=seed, options=options)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/lab/syp/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py", line 296, in reset
    self._reset_idx(indices)
  File "/home/lab/Desktop/ivan/BFM_for_humonoid/Test_training/source/Test_training/Test_training/tasks/direct/test_training/test_training_env.py", line 91, in _reset_idx
    super()._reset_idx(env_ids)
  File "/home/lab/syp/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py", line 595, in _reset_idx
    self.scene.reset(env_ids)
  File "/home/lab/syp/IsaacLab/source/isaaclab/isaaclab/scene/interactive_scene.py", line 450, in reset
    articulation.reset(env_ids)
  File "/home/lab/syp/IsaacLab/source/isaaclab/isaaclab/assets/articulation/articulation.py", line 181, in reset
    self._external_torque_b[env_ids] = 0.0
    ~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^
RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


Set the environment variable HYDRA_FULL_ERROR=1 for a complete stack trace.
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 563
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 566
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 569
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 572
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 575
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 578
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 581
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 584
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 587
2026-01-30T10:20:45Z [9,165ms] [Error] [omni.physx.fabric.plugin] CUDA error: device-side assert triggered: ../../../extensions/runtime/source/omni.physx.fabric/plugins/DirectGpuHelper.cpp: 610
2026-01-30T10:20:45Z [9,166ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,166ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,166ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator' for removal
2026-01-30T10:20:45Z [9,166ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,166ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Annotators' for removal
2026-01-30T10:20:45Z [9,167ms] [Warning] [omni.graph.core.plugin] Could not find category 'Replicator:Core' for removal
2026-01-30T10:20:45Z [9,168ms] [Warning] [omni.graph.core.plugin] Could not find category 'animation' for removal
2026-01-30T10:20:45Z [9,169ms] [Warning] [omni.physx.plugin] USD stage detach not called, holding a loose ptr to a stage!
2026-01-30T10:20:45Z [9,169ms] [Warning] [omni.physx.plugin] PhysX warning: /builds/omniverse/physics/physx/source/gpucommon/src/PxgCudaMemoryAllocator.cpp, FILE /builds/omniverse/physics/physx/source/gpucommon/src/PxgCudaMemoryAllocator.cpp, LINE 68


