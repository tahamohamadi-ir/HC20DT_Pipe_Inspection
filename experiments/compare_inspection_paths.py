import csv
from collections import Counter
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np
from stable_baselines3 import PPO
from simulation.gym_env import PipeInspectionGym
from simulation.reference import fully_outside

policy=PPO.load("models/hc20dt_ppo_4096.zip",device="cpu")
env=PipeInspectionGym()
starts=list(csv.DictReader(Path("evaluation/starts.csv").open()))
rng=np.random.default_rng(1)
fixed=(2,3,5,4,6,7,5,3,1)
short=(2,3,5,4,6,7,1)
for name in ("ppo","reference","direct_exit","uniform"):
    counts=Counter()
    sequences=Counter()
    times=[]
    coverage=[]
    seconds=perf_counter()
    for row in starts * (3 if name=="uniform" else 1):
        offsets=tuple(float(row[k]) for k in ("base_offset_deg","shoulder_offset_deg"))
        obs,_=env.reset(options={"offsets_deg":offsets})
        actions=[]
        while True:
            if name=="ppo":
                a=int(policy.predict(obs,deterministic=True)[0])
            elif name=="uniform":
                a=int(rng.integers(env.action_space.n))
            else:
                a=(fixed if name=="reference" else short)[len(actions)]
            actions.append(a)
            obs,_,terminated,truncated,info=env.step(a)
            if terminated or truncated:
                break
        ins=env.inspection
        local=[sum(band==b for band,_ in ins.covered) for b in (100,200,300)]
        coverage.append(sum(local))
        counts[(info["outcome"],bool(fully_outside(ins.data)))]+=1
        times.append(ins.data.time)
        sequences[tuple(actions)]+=1
        if name!="uniform":
            print("run",name,row["episode"],local,info["outcome"],round(ins.data.time,3),flush=True)
    print("SUMMARY",name,dict(counts),"coverage(min/mean)",min(coverage),round(np.mean(coverage),2),
          "sim_s_min_median_max",*[round(x,3) for x in (min(times),median(times),max(times))],
          "wall_s",round(perf_counter()-seconds,3),"sequences",len(sequences),"nominal",round(times[0],3),
          "seq",sequences.most_common(1),flush=True)
