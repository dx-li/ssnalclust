# Recorded Optical Digits execution

The following launcher was used for the recorded runs. Its absolute working
directory and temporary log paths describe the original host. Each child was
started after the preceding child terminated. Python subprocess timeout handling
kills and waits for the direct worker; the worker does not launch solver subprocesses.
The portable rerun entry point is `examples/run_digits_study.py`.

```python
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
root=Path('/Users/dxli2/math stats/ssnalclust')
env=os.environ.copy()
thread_names=['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS']
env.update({key:'1' for key in thread_names})
records=[]
for k in [10,20]:
    command=[sys.executable,'examples/digits_study.py','--neighbors',str(k),
             '--output',f'docs/digits_k{k}.json','--arrays',f'docs/digits_k{k}.npz']
    started=time.perf_counter()
    print('Starting graph',k,flush=True)
    with open(f'/tmp/ssnalclust-digits-k{k}.log','w') as log:
        try:
            child=subprocess.run(command,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=900)
            status='completed' if child.returncode==0 else 'error';returncode=child.returncode
        except subprocess.TimeoutExpired:
            status='timeout';returncode=None
    record=dict(neighbors=k,process_status=status,returncode=returncode,process_wall_seconds=time.perf_counter()-started,
                timeout_seconds=900,requested_thread_environment={key:env[key] for key in thread_names},
                command=['python',*command[1:]],protocol_commit='9f522e9',implementation_commit='29853ed')
    for suffix in ['json','npz']:
        path=root/f'docs/digits_k{k}.{suffix}'
        record[f'{suffix}_exists']=path.exists()
        if path.exists():record[f'{suffix}_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    records.append(record)
    (root/'docs/digits_processes.json').write_text(json.dumps(records,indent=2)+'\n',encoding='utf-8')
    print('Finished graph',k,status,record['process_wall_seconds'],flush=True)
```

Parent log:

```text
Starting graph 10
Finished graph 10 completed 119.19270229106769
Starting graph 20
Finished graph 20 completed 147.19190154178068
```

Graph 10 worker log:

```text
k=10 SSNAL index=0 gamma=0.00266047 status=completed converged=True
k=10 SSNAL index=1 gamma=0.00665119 status=completed converged=True
k=10 SSNAL index=2 gamma=0.0133024 status=completed converged=True
k=10 SSNAL index=3 gamma=0.0266047 status=completed converged=True
k=10 SSNAL index=4 gamma=0.0665119 status=completed converged=True
k=10 SSNAL index=5 gamma=0.133024 status=completed converged=True
k=10 SSNAL index=6 gamma=0.266047 status=completed converged=True
k=10 SSNAL index=7 gamma=0.665119 status=completed converged=True
k=10 SSNAL index=8 gamma=1.33024 status=completed converged=True
k=10 ADMM index=0 status=completed converged=True
k=10 ADMM index=4 status=completed converged=True
k=10 ADMM index=8 status=completed converged=True
```

Graph 20 worker log:

```text
k=20 SSNAL index=0 gamma=0.00146975 status=completed converged=True
k=20 SSNAL index=1 gamma=0.00367437 status=completed converged=True
k=20 SSNAL index=2 gamma=0.00734874 status=completed converged=True
k=20 SSNAL index=3 gamma=0.0146975 status=completed converged=True
k=20 SSNAL index=4 gamma=0.0367437 status=completed converged=True
k=20 SSNAL index=5 gamma=0.0734874 status=completed converged=True
k=20 SSNAL index=6 gamma=0.146975 status=completed converged=True
k=20 SSNAL index=7 gamma=0.367437 status=completed converged=True
k=20 SSNAL index=8 gamma=0.734874 status=completed converged=True
k=20 ADMM index=0 status=completed converged=True
k=20 ADMM index=4 status=completed converged=True
k=20 ADMM index=8 status=completed converged=True
```
