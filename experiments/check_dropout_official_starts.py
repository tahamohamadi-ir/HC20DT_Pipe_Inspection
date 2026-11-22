import csv
from collections import Counter
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
import simulation.inspection_env as ie
from simulation.gym_env import PipeInspectionGym

env=PipeInspectionGym()
policy=PPO.load("models/hc20dt_ppo_4096.zip",device="cpu")
saved=ie.scan_target
frozen=(2,7,2,5,2,5,2,5,1)
starts=list(csv.DictReader(Path("evaluation/starts.csv").open()))
for row in starts:
    offsets=tuple(float(row[k]) for k in ("base_offset_deg","shoulder_offset_deg"))
    for lost in (2,):
        for mode in ("ppo","replay"):
            obs,_=env.reset(options={"offsets_deg":offsets})
            ins=env.inspection
            decisions=[]
            denied=Counter()
            per_action=[]
            while True:
                a=int(policy.predict(obs,deterministic=True)[0]) if mode=="ppo" else frozen[len(decisions)]
                decisions.append(a)
                def scan(m,d):
                    hit=saved(m,d)
                    if hit is not None and len(decisions)==lost:
                        denied[hit]+=1
                        return None
                    return hit
                ie.scan_target=scan
                obs,_,term,trunc,info=env.step(a)
                per_action.append(len(ins.covered))
                if term or trunc:
                    break
            ie.scan_target=saved
            counts=[sum(b==d for b,_ in ins.covered) for d in (100,200,300)]
            match=(tuple(decisions)==frozen)
            print("episode",row["episode"],"drop_action",lost,"mode",mode,"outcome",info["outcome"],"coverage",counts,
                  "same",match,"decisions",decisions,"covered_after",per_action,"suppressed_targets",len(denied),
                  "sim_time",round(ins.data.time,3),flush=True)
