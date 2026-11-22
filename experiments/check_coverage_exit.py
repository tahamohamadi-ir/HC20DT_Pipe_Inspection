import csv
from collections import Counter
from pathlib import Path
from statistics import median
from stable_baselines3 import PPO
import simulation.inspection_env as ie
from simulation.gym_env import PipeInspectionGym

env=PipeInspectionGym()
ins=env.inspection
policy=PPO.load("models/hc20dt_ppo_4096.zip",device="cpu")
scan=ie.scan_target
starts=list(csv.DictReader(Path("evaluation/starts.csv").open()))
for blind_second in (False,True):
    summary=Counter()
    times=[]
    for row in starts:
        offsets=tuple(float(row[k]) for k in ("base_offset_deg","shoulder_offset_deg"))
        obs,_=env.reset(options={"offsets_deg":offsets})
        actions=[]
        while True:
            if len(ins.covered)==24 and actions:
                level,side=divmod(actions[-1],2)
                assert level>0
                action=2*(level-1)+side
            else:
                action=int(policy.predict(obs, deterministic=True)[0])
            actions.append(action)
            ie.scan_target=(lambda m,d: None) if blind_second and len(actions)==2 else scan
            try:
                obs,_,term,trunc,info=env.step(action)
            finally:
                ie.scan_target=scan
            if term or trunc: break
        times.append(ins.data.time)
        summary[(info["outcome"],len(ins.covered),tuple(actions))]+=1
    print("drop_second",blind_second,"cases",summary,"median_s",round(median(times),3),flush=True)
