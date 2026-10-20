"""Exercise worker IPC and ordering independently of an Isaac installation."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from isaac_rl.chart_cma_distributed import DistributedIsaacEvaluator


@pytest.fixture
def workers(tmp_path, monkeypatch):
    import isaac_rl.chart_cma_distributed as distributed
    fake = tmp_path/'fake_worker.py'
    fake.write_text('''
import sys, time, json, pickle, hashlib
from pathlib import Path
folder=Path(sys.argv[sys.argv.index('--folder')+1])
cfg=json.loads((folder/'setup.json').read_text())
def write(path, value):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value))
    temp.replace(path)
write(folder/'ready.json', {'checkpoint_sha256': hashlib.sha256(Path(cfg['checkpoint']).read_bytes()).hexdigest()})
request=1
while not (folder/'stop').exists():
    key=f'request_{request:06d}'
    path=folder/(key+'.pkl')
    if not path.exists():
        time.sleep(.01)
        continue
    payload=pickle.loads(path.read_bytes())
    if isinstance(payload,dict):
        episodes=payload['episodes'];states=payload['initial_states']
    else:
        episodes=payload;states=None
    time.sleep(.12 if cfg['args']['device']=='cuda:0' else .01)
    if 'fail' in episodes:
        write(folder/(key+'.json'), {'error':'deliberate rollout failure'})
    else:
        reports=[None if x is None else {'episode':x,'reward':x/10} for x in episodes]
        if states is not None:
            for report,state in zip(reports,states):
                if report is not None:report['initial_state']=state
        write(folder/(key+'.json'), {'reports':reports})
    request+=1
''')
    original = distributed.subprocess.Popen
    def launch(command, **kwargs):
        command = list(command)
        command[2] = str(fake)
        return original(command, **kwargs)
    monkeypatch.setattr(distributed.subprocess, 'Popen', launch)
    checkpoint = tmp_path/'decoder.pt'
    checkpoint.write_bytes(b'frozen test decoder')
    evaluator = DistributedIsaacEvaluator(SimpleNamespace(out=str(tmp_path/'run'),
        envs=8, rollout_devices=['cuda:0', 'cuda:1']), checkpoint, tmp_path/'seed.json')
    try:
        yield evaluator
    finally:
        evaluator.close()
    assert all(process.poll() is not None for _, process in evaluator.workers)


def test_async_workers_keep_offspring_order_and_persist_across_generations(workers):
    assert workers.evaluate(list(range(9))) == [dict(episode=x, reward=x/10) for x in range(9)]
    assert workers.evaluate([13]) == [dict(episode=13, reward=1.3)]
    assert workers.evaluate([]) == []
    assert workers.request == 2


def test_worker_failure_prevents_partial_population_result(workers):
    with pytest.raises(RuntimeError, match='deliberate rollout failure'):
        workers.evaluate([0, 1, 'fail', 3])


def test_physical_initial_states_keep_alignment_across_workers_and_waves(workers):
    states = [dict(q=[i],qd=[-i]) for i in range(9)]
    reports = workers.evaluate(list(range(9)), initial_states=states, collect_post_hits=True)
    assert [r['initial_state'] for r in reports] == states
    assert [r['episode'] for r in reports] == list(range(9))
    with pytest.raises(ValueError, match='initial state'):
        workers.evaluate([1,2],initial_states=states[:1])


def test_explicit_idle_slots_keep_neighbor_positions_and_return_missing_labels(workers):
    states=[{'q':[i]} for i in range(4)]
    reports=workers.evaluate([1,None,3,None],initial_states=states)
    assert reports[1] is None and reports[3] is None
    assert reports[0]['initial_state']==states[0] and reports[2]['initial_state']==states[2]
    assert workers.evaluate([None]*4,initial_states=states)==[None]*4


def test_resume_retains_or_explicitly_replaces_rollout_devices(tmp_path):
    from scripts.train_chart_cma import parse_args
    (tmp_path/'config.json').write_text(json.dumps({'args': dict(sampling='independent',
        rollout_devices=['cuda:0', 'cuda:1'], population=4096, envs=4096)}))
    flags = ['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')]
    assert parse_args(flags).rollout_devices == ['cuda:0', 'cuda:1']
    assert parse_args(flags+['--rollout-devices', 'cuda:2', 'cuda:3']).rollout_devices == ['cuda:2', 'cuda:3']
    with pytest.raises(SystemExit):
        parse_args(flags+['--rollout-devices', 'cuda:1', 'cuda:1'])


def test_physics_frequency_changes_only_substeps_and_survives_resume(tmp_path):
    from scripts.train_chart_cma import parse_args, physics_timing
    assert physics_timing(100) == (.01, 2)
    assert physics_timing(250) == (.004, 5)
    for frequency in [0, -100, 120, float('inf'), float('nan')]:
        with pytest.raises(ValueError):
            physics_timing(frequency)
    (tmp_path/'config.json').write_text(json.dumps({'args': dict(sampling='independent', physics_hz=250)}))
    flags = ['--out', str(tmp_path), '--resume', str(tmp_path/'last.pkl')]
    assert parse_args(flags).physics_hz == 250
    assert parse_args(flags+['--physics-hz', '100']).physics_hz == 100
    (tmp_path/'config.json').write_text(json.dumps({'args': dict(sampling='independent', physics_hz=100)}))
    assert parse_args(flags).physics_hz == 100
