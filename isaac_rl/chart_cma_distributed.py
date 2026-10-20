"""Persistent, isolated Isaac processes; one central CMA population update."""
from __future__ import annotations

import json
import hashlib
import math
import os
from pathlib import Path
import pickle
import subprocess
import sys
import time


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def split_rollouts(episodes, workers):
    """Contiguous shards with stable original report ordering, including empties."""
    return [episodes[len(episodes)*i//workers:len(episodes)*(i+1)//workers]
            for i in range(workers)]


class DistributedIsaacEvaluator:
    def __init__(self, args, checkpoint, seed_clip):
        self.workers = []
        self.request = 0
        root = Path(__file__).resolve().parents[1]
        session = Path(args.out)/'rollout_workers'/f'{time.time_ns()}_{os.getpid()}'
        self.session = session
        capacity = math.ceil(args.envs/len(args.rollout_devices))
        try:
            for index, device in enumerate(args.rollout_devices):
                folder = session/f'worker_{index}'
                folder.mkdir(parents=True)
                settings = dict(vars(args), device=device, envs=capacity)
                atomic_json(folder/'setup.json', dict(args=settings,
                    checkpoint=str(Path(checkpoint).resolve()), seed_clip=str(Path(seed_clip).resolve())))
                log = (folder/'worker.log').open('a')
                try:
                    process = subprocess.Popen([sys.executable, '-u',
                        str(root/'scripts/chart_cma_rollout_worker.py'), '--folder', str(folder.resolve())],
                        cwd=root, stdout=log, stderr=subprocess.STDOUT,
                        env=dict(os.environ, OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1'))
                finally:
                    log.close()
                self.workers.append((folder, process))
            deadline = time.monotonic()+300
            while True:
                ready = sum((folder/'ready.json').exists() for folder, _ in self.workers)
                if ready == len(self.workers):
                    break
                self._check_workers()
                if time.monotonic() > deadline:
                    raise TimeoutError(f'Isaac workers not ready; inspect {session}')
                time.sleep(.2)
            expected = hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest()
            for folder, _ in self.workers:
                if json.loads((folder/'ready.json').read_text())['checkpoint_sha256'] != expected:
                    raise ValueError('Rollout worker loaded a different frozen decoder')
            atomic_json(Path(args.out)/'rollout_workers.json', dict(session=str(session.resolve()),
                environments_per_worker=capacity, total_capacity=capacity*len(self.workers),
                workers=[dict(device=device, pid=process.pid, folder=str(folder.resolve()))
                    for device, (folder, process) in zip(args.rollout_devices, self.workers)]))
            print(f'[cma-setup] {len(self.workers)} Isaac workers ready, {capacity} environments each', flush=True)
        except BaseException:
            self.close()
            raise

    def _check_workers(self):
        for folder, process in self.workers:
            if process.poll() is not None:
                raise RuntimeError(f'Isaac worker exited {process.returncode}; inspect {folder}/worker.log')

    def evaluate(self, episodes, progress=None, *, initial_states=None, collect_post_hits=False,
                 collect_prefirst_snapshots=False, command_overrides=None):
        if not episodes:
            return []
        if initial_states is not None and len(initial_states) != len(episodes):
            raise ValueError('One physical initial state per episode required')
        if command_overrides is not None and len(command_overrides) != len(episodes):
            raise ValueError('One command override entry per episode required')
        self.request += 1
        key = f'request_{self.request:06d}'
        shards = split_rollouts(episodes, len(self.workers))
        state_shards = split_rollouts(initial_states, len(self.workers)) if initial_states is not None else [None]*len(self.workers)
        override_shards = (split_rollouts(command_overrides, len(self.workers))
                           if command_overrides is not None else [None]*len(self.workers))
        for (folder, _), shard, states, overrides in zip(
                self.workers, shards, state_shards, override_shards):
            target = folder/(key+'.pkl')
            temporary = target.with_suffix('.tmp')
            payload = (dict(episodes=shard, initial_states=states, collect_post_hits=collect_post_hits,
                collect_prefirst_snapshots=collect_prefirst_snapshots,
                command_overrides=overrides)
                if initial_states is not None or collect_post_hits or collect_prefirst_snapshots
                or command_overrides is not None else shard)
            temporary.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
            temporary.replace(target)
        pending = set(range(len(self.workers)))
        results = [None]*len(self.workers)
        last_progress = None
        while pending:
            for i in list(pending):
                folder, _ = self.workers[i]
                response = folder/(key+'.json')
                if response.exists():
                    result = json.loads(response.read_text())
                    if 'error' in result:
                        raise RuntimeError(f'Isaac worker {i}: {result["error"]}; inspect {folder}/worker.log')
                    if len(result['reports']) != len(shards[i]):
                        raise ValueError('Worker report count differs from assigned offspring count')
                    results[i] = result['reports']
                    pending.remove(i)
            if pending:
                self._check_workers()
            if progress:
                states = []
                for i, (folder, _) in enumerate(self.workers):
                    path = folder/(key+'_progress.json')
                    states.append(json.loads(path.read_text()) if path.exists() else dict(tick=0, ticks=0))
                current = tuple((s['tick'], s['ticks']) for s in states)
                if current != last_progress:
                    atomic_json(self.session/'progress.json', dict(request=self.request,
                        workers=[dict(index=i, completed=i not in pending, **state) for i, state in enumerate(states)]))
                    progress(sum(len(shards[i]) for i in range(len(shards)) if i not in pending),
                        len(episodes), min(s['tick'] for s in states), max(s['ticks'] for s in states))
                    last_progress = current
            if pending:
                time.sleep(.1)
        return [report for shard in results for report in shard]

    def close(self):
        for folder, _ in self.workers:
            (folder/'stop').touch()
        for folder, process in self.workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.terminate()
        for _, process in self.workers:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
