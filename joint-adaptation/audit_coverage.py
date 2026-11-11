"""Audit carrier assignments, optimizer endpoints, and late-stage changes."""

import argparse
from collections import Counter
import hashlib
from pathlib import Path

import numpy as np
import torch

from analyze_probe import key, load, load_lines, paired, sha256
from rgwm.cli.common import make_dataset
from rgwm.config import load_run_config
from run_probe import write_json


def digest(state):
    result = hashlib.sha256()
    for name, tensor in state.items():
        result.update(name.encode())
        result.update(tensor.contiguous().numpy().tobytes())
    return result.hexdigest()


def audit_checkpoints(folder, meta, parent_hashes):
    buffers = ('running_mean', 'running_var', 'num_batches_tracked')
    initial_buffers = {}
    for step, expected in ((0, meta['start_hashes']), (6000, meta['end_hashes'])):
        ckpt = folder / 'checkpoints' / f'step-{step:04d}.pt'
        if step:
            assert sha256(ckpt) == meta['final_checkpoint_sha256']
        state = torch.load(ckpt, map_location='cpu', weights_only=True)
        assert state['step'] == step and state['data_cursor'] == step*16
        opt = state['optimizer']
        groups = opt['param_groups']
        assert all(g['lr'] == 5e-6 and g['weight_decay'] == 1e-5 for g in groups)
        expected_slots = sum(len(g['params']) for g in groups) if step else 0
        assert len(opt['state']) == expected_slots
        assert all(int(slot['step']) == step for slot in opt['state'].values())
        for name in ('encoder', 'decoder'):
            hashed = digest(state[name])
            assert hashed == expected[name]
            stats = digest({k:v for k,v in state[name].items() if k.endswith(buffers)})
            if step == 0:
                assert hashed == parent_hashes[name]
                initial_buffers[name] = stats
            else:
                assert initial_buffers[name] == stats
        del state


def audit_pool(plan, index, count):
    pool = [index[sample] for sample in plan['pools'][str(count)]]
    by_identity = Counter(r['identity_id'] for r in pool)
    groups = Counter((r['identity_id'],r['group_id']) for r in pool)
    exposures = Counter()
    for batch in plan['batches']:
        indices = batch['indices'][str(count)]
        exposures.update(indices)
        for offset in (0,8):
            assert Counter(pool[i]['identity_id'] for i in indices[offset:offset+8]) == {'cage':4,'trump':4}
    audit = {}
    for identity in ('cage','trump'):
        uses = [exposures[i] for i,r in enumerate(pool) if r['identity_id'] == identity]
        assert len(uses) == by_identity[identity] and sum(uses) == 48000 and max(uses)-min(uses) <= 1
        audit[identity] = dict(covers=len(uses), groups=sum(k[0] == identity for k in groups),
                               exposure_range=[min(uses),max(uses)])
    return audit


def late_stage(folder):
    train = load_lines(folder / 'train.jsonl')
    assert len(train) == 6000
    data = {step:load_lines(folder / f'validation-{step}.jsonl') for step in (4000,5000,5500,6000)}
    result = {}
    for name in ('same','new'):
        rows = {step:sorted((r for r in values if r['panel'] == name and r['path'] == 'swap'), key=key)
                for step,values in data.items()}
        result[name] = {f'{start}_to_6000':paired(rows[start], rows[6000]) for start in (4000,5000,5500)}
    result['train_swap_bce'] = {}
    for step in (5000,5500,5800,5900,6000):
        for window in (100,500):
            result['train_swap_bce'][f'{step-window+1}-{step}'] = float(np.mean([
                row[slot]['processed_bce'] for row in train[step-window:step] for slot in ('designated','common')]))
    return result


def main(args):
    plan = load(args.root / 'shared-plan.json')
    cfg = load_run_config(args.config,args.assets.resolve())
    assert sha256(Path(cfg['assets']['manifest'])) == plan['manifest_sha256']
    records = make_dataset(cfg,split='train').records
    index = {r['sample_id']:r for r in records}
    assert len(index) == 416 and set(index) == set(plan['pools']['416'])
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    assert state['metadata']['steps'] == 500 and state['metadata']['train_cover_count'] == 416
    parent_hashes = {name:digest(state[name]) for name in ('encoder','decoder')}
    parent_file = sha256(args.checkpoint)
    del state
    pools, trajectory, audit = {}, {}, []
    for n in (16,64,416):
        tag = f'n{n}'
        folder = args.root / f'{tag}-seed0'
        meta = load(folder / 'complete.json')
        assert meta['checkpoint_sha256'] == parent_file and meta['test_images_read'] == 0
        audit_checkpoints(folder,meta,parent_hashes)
        pools[tag] = audit_pool(plan,index,n)
        trajectory[tag] = late_stage(folder)
        audit.append(f'{tag} verified shared start, frozen BN buffers, fresh Adam, 6000 Adam steps, cursor 96000')
        print(audit[-1], flush=True)
    write_json(args.root / 'pools-audit.json', pools)
    write_json(args.root / 'late-stage.json', trajectory)
    (args.root / 'checkpoint-audit.txt').write_text('\n'.join(audit)+'\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','config','assets','checkpoint'):
        parser.add_argument('--'+name,type=Path,required=True)
    main(parser.parse_args())
