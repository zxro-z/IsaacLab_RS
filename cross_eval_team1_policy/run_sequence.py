import os,sys,subprocess,json,shlex
from pathlib import Path
root=Path('/home/zxro/teammate_ant_rl');out=root/'cross_eval_team1_policy'
env=dict(os.environ,ISAAC_SIM_SITE_PACKAGES='/home/zxro/anaconda3/envs/env_isaaclab/lib/python3.11/site-packages',PYTHONDONTWRITEBYTECODE='1',CONDA_PREFIX='/home/zxro/anaconda3/envs/env_isaaclab_231')
python='/home/zxro/anaconda3/envs/env_isaaclab_231/bin/python'
stages=[('final_unseen',4,'smoke'),('teammate2',4,'smoke'),('final_unseen',100,'main'),('teammate1',100,'main'),('teammate2',100,'main')]
for host,n,stage in stages:
    target=out/host/stage;log=out/host/(stage+'.log');log.parent.mkdir(exist_ok=True,parents=True)
    if target.exists() or log.exists():raise FileExistsError(target)
    cmd=[python,'-u','cross_eval_team1_policy/evaluate_depth_policy.py','--environment',host,'--seed','24','--num_envs',str(n),'--headless','--output',str(target)]
    with (out/'commands.log').open('a') as f:f.write('ISAAC_SIM_SITE_PACKAGES='+env['ISAAC_SIM_SITE_PACKAGES']+' PYTHONDONTWRITEBYTECODE=1 CONDA_PREFIX='+env['CONDA_PREFIX']+' '+shlex.join(cmd)+'\n')
    print('START '+host+' '+stage,flush=True)
    with log.open('w') as f:
        process=subprocess.Popen(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT)
        # Result is written before simulator shutdown, which can delay on this runtime.
        name='smoke.json' if n==4 else 'evaluation_seed24_n100.json'
        try: code=process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:process.wait(timeout=10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
            raise RuntimeError('Run timeout; see '+str(log))
    if process.returncode!=0:raise RuntimeError('Run failed; see '+str(log))
    result=json.loads((target/name).read_text())
    assert result['completed']==n and result['identity']['all_finite']
    print('PASS '+host+' '+stage+' '+json.dumps(result['summary']['episode_return']),flush=True)
print('ALL STAGES COMPLETE',flush=True)
