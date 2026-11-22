# training

All training runs use PPO, DQN or A2C from Stable-Baselines3 on the environment in `simulation/gym_env.py`. Run the commands from the repository root.

| File | What it does |
|---|---|
| `pilot.py` | Trains PPO for 128 decisions with seed 0 and checks it on the five validation starts. This is the first half of the submitted recipe. |
| `checkpoints.py` | Loads the pilot and continues to 1,024, 2,048 and 4,096 decisions, checking each checkpoint. The 4,096 checkpoint became `models/hc20dt_ppo_4096.zip`. |
| `seed_study.py` | Trains PPO from scratch for seeds 0–4 without the pilot restart and checks it at five budgets up to 8,192 decisions. |
| `compare_controllers.py` | Trains PPO, DQN and A2C with the same settings and evaluates them next to the scripted controllers. |
| `compare_devices.sh` | Runs that comparison with training and evaluation on both CPU and CUDA, and saves a log of the run. |

To repeat the submitted training:

```bash
python -m training.pilot && python -m training.checkpoints
```

Both scripts save their models in `/tmp`. Copy `/tmp/hc20dt_ppo_4096.zip` to `models/` if you want to replace the submitted policy.

Training times, the seed study and the algorithm comparison are described in the [main README](../README.md#train-the-policy).
