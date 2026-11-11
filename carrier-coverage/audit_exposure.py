"""Recompute exposure-run endpoints and apply the predeclared N416 gate."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from analyze_joint import PATHS, contrasts, validation
from analyze_probe import load, load_lines, sha256
from exposure_plan import MILESTONES, START, prior_history
from rgwm.data.messages import split_message
from run_probe import write_json


def audit_training(folder, through, size, carriers, parent, indices, codes, schedule, weights,
                   *, first_step=START, ofea_weight=None):
    history, seen, frequencies = prior_history(parent, range(size))
    assert history['train_ids'] == schedule['train_ids']
    end = through-START
    selected, messages = indices[:end], codes[:end]
    assert not seen[selected,messages].any()
    encoded = (selected.astype(np.int64) << 15) + messages
    assert np.unique(encoded).size == encoded.size
    for code in set(map(int,np.unique(messages))):
        assert split_message([int(x) for x in f'{code:015b}']) == 'train' and code in frequencies
    exposures = {}
    for step in MILESTONES[size]:
        if step <= through:
            counts = np.bincount(selected[:step-START].ravel(),minlength=size) + seen.sum(axis=1)
            expected = 16*step//size
            assert counts.min() == counts.max() == expected
            exposures[step] = int(expected)
    cage = np.asarray([x == 'cage' for x in carriers])
    log = folder / 'train.jsonl'
    train_means, windows = [], Counter()
    step = first_step
    with log.open(encoding='utf-8') as stream:
        for step, line in enumerate(stream,first_step+1):
            row = json.loads(line)
            assert step == row['step'] <= through and np.isfinite(row['combined_grad_norm']) and row['combined_grad_norm'] > 0
            batch = selected[step-START-1]
            c = int(cage[batch].sum())
            assert row['identity_counts'] == {'cage':c,'trump':16-c}
            for offset, slot in ((0,'designated'),(8,'common')):
                micro = row[slot]
                count = int(cage[batch[offset:offset+8]].sum())
                assert micro['identity_weights'] == {'cage':count/(2*c),'trump':(8-count)/(2*(16-c))}
                assert micro['processed_path'] == 'swap' and 0 <= micro['max_linf'] <= 8/255+1e-7
                expected = (micro['mse'] + .1*micro['structure'] + weights['direct']*micro['direct_bce']
                            + weights['swap']*micro['swap_bce'])
                if ofea_weight is not None:
                    assert np.isfinite(micro['ofea']) and micro['ofea'] >= 0
                    assert np.isclose(micro['weighted_ofea'],ofea_weight*micro['ofea'],rtol=1e-6,atol=1e-9)
                    expected += micro['weighted_ofea']
                    windows['ofea'] += micro['ofea']
                    windows['weighted_ofea'] += micro['weighted_ofea']
                assert abs(expected-micro['contribution']) < 1e-6
                windows['swap_bce'] += micro['swap_bce']
                windows['direct_bce'] += micro['direct_bce']
            if step % 500 == 0:
                train_means.append(dict(step=step, **{k:v/500 for k,v in windows.items()}))
                windows.clear()
    assert step == through
    return exposures, train_means


def read_predictions(folder, step, summaries):
    return validation(folder, step, summaries['panels'], dict.fromkeys(PATHS,PATHS['same']))


def main(args):
    size, end = args.carriers, args.through_step
    path = args.run_dir or args.root / f'n{size}-seed0'
    parent = args.parent / f'n{size}-seed0'
    final = load(path / f'final-{end}.json')
    meta = load(parent / 'complete.json')
    ckpt = path / f'step-{end}.pt'
    schedule = load(args.root / f'n{size}-schedule.json')
    plan = args.root / f'n{size}-schedule.npz'
    assert schedule['plan_sha256'] == final['plan_sha256'] == sha256(plan)
    assert final['step'] == end and final['data_cursor'] == end*16 and final['test_images_read'] == 0
    assert final['message_weights'] == meta['message_weights_per_update']
    assert final['parent_sha256'] == meta['final_checkpoint_sha256'] and final['end_hashes']['oldpan'] == meta['end_hashes']['oldpan']
    assert final['final_checkpoint_sha256'] == sha256(ckpt)
    first_step = final.get('fork_step',START)
    if 'fork_step' in final:
        source = Path(final['fork_checkpoint'])
        assert final['fork_checkpoint_sha256'] == sha256(source)
        assert final['message_weights'] == {'direct':.5,'swap':.5}
        assert source.name == f'step-{first_step}.pt' and first_step < end
        baseline_folder = source.parent
        nodes = final['nodes']
        assert nodes == sorted(set(nodes)) and nodes[0] == first_step and nodes[-1] == end
    else:
        baseline_folder = parent
        nodes = [START]+[step for step in MILESTONES[size] if step <= end]
    state = torch.load(ckpt, weights_only=True, map_location='cpu')
    assert state['step'] == end and state['data_cursor'] == end*16
    assert state['log_offset'] == (path / 'train.jsonl').stat().st_size
    assert state['optimizer']['state'] and all(int(s['step']) == end for s in state['optimizer']['state'].values())
    del state
    manifest = load(args.root / 'carrier-identities.json')
    assert manifest['manifest_sha256'] == final['manifest_sha256']
    identities = [manifest['identities'][i] for i in schedule['train_ids']]
    with np.load(plan, allow_pickle=False) as data:
        exposure, averages = audit_training(path,end,size,identities,parent,data['carriers'],data['messages'],
                                            schedule,final['message_weights'],first_step=first_step,
                                            ofea_weight=final.get('ofea_weight'))
    assert sha256(path / f'validation-{first_step}.jsonl') == sha256(baseline_folder / f'validation-{first_step}.jsonl')
    summaries = [load(path / f'evaluation-{step}.json') for step in nodes]
    predictions = {step:read_predictions(path,step,summary) for step,summary in zip(nodes,summaries,strict=True)}
    initial = predictions[first_step]
    against_initial, against_previous = {}, {}
    for previous, step in zip(nodes,nodes[1:]):
        current = predictions[step]
        against_initial[step] = {name:contrasts(initial[name]['swap'],current[name]['swap']) for name in PATHS}
        against_previous[step] = {name:contrasts(predictions[previous][name]['swap'],current[name]['swap']) for name in PATHS}
    against_matched = {}
    comparison = {64:(24000,16),416:(39000,64)}[size]
    for matched_step, control in (comparison,(156000,16)) if size == 416 else (comparison,):
        if matched_step in predictions:
            control_dir = args.parent / f'n{control}-seed0'
            rows = read_predictions(control_dir,START,load(control_dir / 'evaluation.json')[-1])
            against_matched[matched_step] = dict(control=f'N{control} at step {START}',
                **{name:contrasts(rows[name]['swap'],predictions[matched_step][name]['swap']) for name in PATHS})
    result = dict(size=size, through=end, cumulative_exposures=exposure, trajectory=summaries,
                  versus_start=against_initial, versus_previous=against_previous,
                  matched_exposure=against_matched, training_500_step_means=averages)
    if 'identity_pairs_sha256' in final:
        from exposure_identity import identity_summary
        clean_controls = None
        for step,summary in zip(nodes,summaries,strict=True):
            identity_rows = load_lines(path / f'identity-{step}.jsonl')
            message_rows = predictions[step]['new']['swap']
            fields = ('sample_id','message_index','bit_errors')
            assert sorted(tuple(r[k] for k in fields) for r in identity_rows) == sorted(
                tuple(r[k] for k in fields) for r in message_rows)
            assert summary['identity'] == identity_summary(identity_rows)
            current = {r['sample_id']:(r['source_sample_id'],r['clean_eligible'],r['clean_identity_score'])
                       for r in identity_rows}
            assert all(current[r['sample_id']] == (r['source_sample_id'],r['clean_eligible'],r['clean_identity_score'])
                       for r in identity_rows)
            if clean_controls is None:
                clean_controls = current
            assert current == clean_controls
    if size == 416 and end >= 39000 and 'fork_step' not in final:
        lower = [against_initial[step]['new']['all']['wmr_ci95'][0] for step in (26000,39000)]
        counts = [sum(r['bit_errors']==0 for r in predictions[step]['new']['swap']) for step in (26000,39000)]
        result['gate'] = dict(continue_to_156000=all(x>0 for x in lower) and counts[1]>=counts[0],
                              reference=START, assessed_steps=[26000,39000], exact_counts=counts,
                              paired_wmr_ci95_lower=lower,
                              rule='Both paired WMR 95% lower bounds versus step 6000 above zero and step 39000 count at least step 26000 count')
    write_json(path / f'audit-{end}.json',result)
    print(json.dumps(dict(steps=end, new_swap=summaries[-1]['panels']['new']['swap']['all'],
                          gate=result.get('gate'))), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root','parent'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--carriers',type=int,choices=tuple(MILESTONES),required=True)
    parser.add_argument('--through-step',type=int,required=True)
    parser.add_argument('--run-dir',type=Path,help='Specific fork output directory to audit')
    main(parser.parse_args())
