"""Run budget-matched multi-seed optimizer comparisons on the M5 Mac."""

from __future__ import annotations

import argparse, csv, json, platform, time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset

from wipe_esn_experiment import ESN, fit_bc_initializer, pack_episodes, rollout
from wipe_optimizers import cem, random_search, spsa_adapter


def main():
    p=argparse.ArgumentParser(); p.add_argument("--budget",type=int,default=8)
    p.add_argument("--seeds",default="0,1,2"); p.add_argument("--output",type=Path,default=Path("../results/main_independent_esn/mac_optimizer_benchmark.json"))
    a=p.parse_args(); seeds=[int(x) for x in a.seeds.split(',')]
    root=Path(__file__).resolve().parent.parent
    mjcf=root/'unitree_mujoco/unitree_robots/g1/g1_29dof.xml'
    ds=load_dataset('unitreerobotics/G1_Dex1_Wipe_Table')['train']
    eps=pack_episodes(ds,[0,1,2,3,160,161,162])
    methods={'random':random_search,'spsa':spsa_adapter,'cem':cem}; rows=[]
    started=time.perf_counter()
    for seed in seeds:
        base=ESN(116,seed=seed); fit_bc_initializer(base,{i:eps[i] for i in (0,1,2,3)})
        base_w=base.Wout.copy()
        for name,fn in methods.items():
            base.Wout=base_w.copy()
            _,train,history=fn(base,eps[0],mjcf,budget=a.budget,seed=seed)
            for heldout_ep in (160,161,162):
                test=rollout(base,eps[heldout_ep],mjcf,teacher_weight=0.0)
                rows.append({
                    'seed':seed,'method':name,'budget':a.budget,'heldout_episode':heldout_ep,
                    'train_L_task':train['L_task'],'heldout_L_task':test['L_task'],
                    'grasp_success':test['grasp_success'],'task_success':test['task_success'],
                    'wipe_path_m':test['wipe_path_length_m'],'contact_ratio':test['table_contact_ratio'],
                    'coverage_m2':test['wipe_coverage_m2'],'teacher_cache':'absent_task_only',
                })
            ckpt=a.output.parent/f'esn_{name}_seed{seed}.npz'; ckpt.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(ckpt,Wout=base.Wout,Win=base.Win,W=base.W,mean=base.mean,scale=base.scale)
    summary={
        'schema':'independent_esn_mac_optimizer_benchmark_v1','created_utc':datetime.now(timezone.utc).isoformat(),
        'host':{'platform':platform.platform(),'chip':'Apple M5','memory_gb':32,'torch':torch.__version__,'mps':torch.backends.mps.is_available()},
        'seeds':seeds,'methods':list(methods),'budget_per_method_seed':a.budget,
        'rollout_evaluations_minimum':len(seeds)*len(methods)*a.budget,
        'teacher_status':'Real UnifoLM cache unavailable on Mac; comparison is task-only and not the final teacher-guided result.',
        'elapsed_s':time.perf_counter()-started,'rows':rows,
    }
    a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(summary,indent=2),encoding='utf-8')
    csv_path=a.output.with_suffix('.csv')
    with csv_path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps({k:v for k,v in summary.items() if k!='rows'},indent=2)); print(csv_path)


if __name__=='__main__': main()
