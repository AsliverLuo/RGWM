"""Verify all joint-adaptation rows and compare matched controls."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np

from analyze_probe import key, load, load_lines, paired, sha256, subset


NODES = [0, 100, 250, 500, 750, 1000]
LATER_NODES = list(range(0, 2001, 250))
COVERAGE_NODES = [0, 250, 500, 1000, 2000, 3000, 4000, 5000, 5500, 6000]
SCOPES = ('all', 'cage', 'trump')
PATHS = {'same': ('swap', 'clean_swap', 'direct', 'png', 'jpeg50'),
         'new': ('swap', 'clean_swap', 'direct')}
COUNTS = {'same': (512, 256, 256), 'new': (556, 256, 300)}


def validation(directory, step, summaries, paths_by_panel):
    result = {panel: {path: {} for path in paths} for panel, paths in paths_by_panel.items()}
    for row in load_lines(directory / f'validation-{step:04d}.jsonl'):
        indexed = result[row['panel']][row['path']]
        assert key(row) not in indexed
        assert len(row['predicted_bits']) == len(row['message']) == 15
        assert set(row['predicted_bits'] + row['message']) <= {0, 1}
        assert row['bit_errors'] == sum(a != b for a, b in zip(row['predicted_bits'], row['message'], strict=True))
        assert 0 <= row['linf'] <= 8/255 + 1e-7
        assert 0 <= row['budget_saturated'] <= 1 and 0 <= row['candidate_clipped'] <= 1
        indexed[key(row)] = row
    for panel, paths in result.items():
        order = sorted(paths['swap'])
        assert len(order) == COUNTS[panel][0]
        n_images, per_image = (16, 32) if panel == 'same' else (139, 4)
        counts = Counter(sample for sample, _ in order)
        assert len(counts) == n_images and set(counts.values()) == {per_image}
        truth = {k: row['message'] for k, row in paths['swap'].items()}
        for path, rows in paths.items():
            assert sorted(rows) == order and all(truth[k] == row['message'] for k, row in rows.items())
            paths[path] = [rows[k] for k in order]
            for scope, n in zip(SCOPES, COUNTS[panel], strict=True):
                errors = [r['bit_errors'] for r in subset(paths[path], scope)]
                expected = summaries[panel][path][scope]
                assert n == len(errors) == expected['n'] and expected['exact'] == errors.count(0)
                assert abs(expected['ber'] - sum(errors)/(15*n)) < 1e-12
                assert expected['error_buckets'] == dict(zero=errors.count(0), one=errors.count(1), two=errors.count(2),
                                                        three_or_more=sum(e >= 3 for e in errors))
            if path == 'clean_swap':
                per_cover = defaultdict(set)
                for row in paths[path]:
                    per_cover[row['sample_id']].add(tuple(row['predicted_bits']))
                assert all(len(bits) == 1 for bits in per_cover.values()), 'Clean swap varies with the label'
        stats = summaries[panel]['quality']
        direct = paths['direct']
        assert stats['max_linf'] == max(r['linf'] for r in direct)
        for expected, field in (('mean_psnr', 'psnr'), ('mean_budget_saturated', 'budget_saturated'),
                                ('mean_candidate_clipped', 'candidate_clipped')):
            assert abs(stats[expected] - np.mean([r[field] for r in direct])) < 1e-10
    return result


def verify_plan(directory, initial, meta, parent_plan=None):
    steps, start = meta['steps'], meta.get('resume_from_step', 0)
    plan = load(directory / 'plan.json')
    assert len(set(plan['train_ids'])) == 16 and len(plan['batches']) == steps
    by_cover = {r['sample_id']: r['target_id'] for r in initial['same']['swap']}
    assert set(by_cover) == set(plan['train_ids'])
    assert {tuple(r['message']) for r in initial['same']['swap']} == set(map(tuple, plan['same_messages']))
    train_messages, continued_messages, historical_pairs = set(), set(), set()
    message_uses, tiers = Counter(), Counter()
    exposures = Counter()
    entered_recycled = False
    for step, batch in enumerate(plan['batches']):
        assert {micro['slot'] for micro in batch['micros']} == {'common', 'designated'}
        assert sorted(i for micro in batch['micros'] for i in micro['indices']) == list(range(16))
        batch_messages = set()
        for micro in batch['micros']:
            labels = list(map(tuple, micro['message']))
            assert len(set(labels)) == 8
            batch_messages.update(labels)
            pairs = set(zip(micro['indices'], labels, strict=True))
            if start >= 2000:
                assert pairs.isdisjoint(historical_pairs)
                historical_pairs.update(pairs)
            elif start and step >= start:
                assert pairs.isdisjoint(historical_pairs)
            else:
                historical_pairs.update(pairs)
            assert Counter(by_cover[plan['train_ids'][i]] for i in micro['indices']) == {'cage': 4, 'trump': 4}
            exposures.update((micro['slot'], i) for i in micro['indices'])
        assert len(batch_messages) == 16 and all(tuple(1-bit for bit in m) in batch_messages for m in batch_messages)
        if start and step >= start:
            assert all(int.from_bytes(hashlib.sha256(''.join(map(str,m)).encode('ascii')).digest(), 'big') % 10 >= 2
                       for m in batch_messages)
            if start >= 2000:
                if step == start:
                    tiers = Counter(message_uses.values())
                assert batch_messages <= train_messages
                remaining, expected_tiers = 16, Counter()
                for level, count in sorted(tiers.items()):
                    used = min(remaining, count)
                    if used:
                        expected_tiers[level] = used
                        remaining -= used
                actual_tiers = Counter(message_uses[m] for m in batch_messages)
                assert remaining == 0 and actual_tiers == expected_tiers, 'Uneven message reuse'
                for m in batch_messages:
                    tiers[message_uses[m]] -= 1
                    message_uses[m] += 1
                    tiers[message_uses[m]] += 1
            else:
                assert batch_messages.isdisjoint(continued_messages)
                if entered_recycled:
                    assert batch_messages <= train_messages
                entered_recycled |= bool(batch_messages & train_messages)
            continued_messages.update(batch_messages)
        else:
            if start < 2000 or step < 1000:
                assert batch_messages.isdisjoint(train_messages)
            train_messages.update(batch_messages)
            message_uses.update(batch_messages)
    assert all(abs(2*exposures[slot, i] - steps) <= 1 for slot in ('common', 'designated') for i in range(16))
    assert sum(message_uses.values()) == 16*(steps if start >= 2000 else start or steps)
    assert (train_messages | continued_messages).isdisjoint(map(tuple, plan['same_messages']))
    if start:
        assert dict(plan, batches=plan['batches'][:start]) == parent_plan
        if start >= 2000:
            assert meta['continuation_observations'] == 16*(steps-start)
            assert len(historical_pairs) == 16*steps
            assert {str(k): v for k, v in Counter(message_uses.values()).items()} == meta['message_frequency_histogram']
        else:
            assert len(continued_messages) == 16*(steps-start)
        assert len(continued_messages-train_messages) == meta['continuation_new_messages']
        assert len(continued_messages & train_messages) == meta['continuation_recycled_messages']


def verify_message_recipe(meta, train):
    recipe = meta['message_recipe']
    designated = 'geometry_color_clamp' if meta['arm'] == 'j2' else 'swap'
    paths = {'common': 'distortion', 'designated': designated} if recipe == 'mixed' else dict.fromkeys(('common', 'designated'), 'swap')
    direct, processed = {'mixed': (.75, .25), 'swap-weighted': (.75, .25), 'swap-equal': (.5, .5)}[recipe]
    assert meta['processed_paths'] == paths and meta['processed_weight'] == processed
    weights = {'direct': direct}
    for path in paths.values():
        weights[path] = weights.get(path, 0.) + processed/2
    assert meta['message_weights_per_update'] == weights
    for row in train:
        for slot, path in paths.items():
            micro = row[slot]
            assert micro['processed_path'] == path
            expected = micro['mse'] + .1*micro['structure'] + direct*micro['direct_bce'] + processed*micro['processed_bce']
            assert abs(micro['loss'] - expected) < 1e-6


def verify_coverage_plan(directory, initial, meta, shared_path):
    plan, shared = load(directory / 'plan.json'), load(shared_path)
    count = str(meta['carrier_count'])
    assert meta['coverage_plan_sha256'] == sha256(shared_path)
    assert all(meta[k] == shared[k] for k in ('seed', 'steps', 'manifest_sha256', 'design_sha256'))
    pools = shared['pools']
    assert {k:len(set(v)) for k,v in pools.items()} == {'16':16,'64':64,'416':416}
    assert pools['16'] == pools['64'][:16] == pools['416'][:16] and set(pools['64']) < set(pools['416'])
    assert plan['train_ids'] == pools[count]
    same = initial['same']['swap']
    assert {r['sample_id'] for r in same} == set(pools['16'])
    assert not {r['sample_id'] for r in initial['new']['swap']} & set(pools['416'])
    assert {tuple(r['message']) for r in same} == set(map(tuple, plan['same_messages']))
    assert plan['same_messages'] == shared['same_messages']
    assert len(plan['batches']) == len(shared['batches']) == meta['steps']
    pairs, exposures, messages = set(), Counter(), Counter()
    for group, common in zip(plan['batches'], shared['batches'], strict=True):
        slots = {m['slot']: m for m in group['micros']}
        labels = slots['designated']['message'] + slots['common']['message']
        indices = slots['designated']['indices'] + slots['common']['indices']
        assert labels == common['message'] and indices == common['indices'][count]
        assert len(set(indices)) == len({tuple(m) for m in labels}) == 16
        bits = set(map(tuple, labels))
        assert all(tuple(1-b for b in m) in bits for m in bits)
        pairs.update(zip(indices, map(tuple, labels), strict=True))
        exposures.update(indices)
        messages.update(bits)
    assert len(exposures) == int(count) and len(pairs) == 16*meta['steps']
    assert all(int.from_bytes(hashlib.sha256(''.join(map(str,m)).encode('ascii')).digest(), 'big') % 10 >= 2
               for m in messages)
    assert not set(messages) & set(map(tuple, plan['same_messages']))
    assert {str(k):v for k,v in Counter(messages.values()).items()} == shared['message_frequency_histogram']


def verify_run(directory, parent_dir=None, shared_plan=None):
    meta = load(directory / 'complete.json')
    steps = meta['steps']
    start_step = meta.get('resume_from_step', 0)
    limit = 6000 if shared_plan else (2000 if start_step >= 2000 else 1000)
    assert start_step in (0, 1000, 2000) and 1 <= steps-start_step <= limit
    assert bool(start_step) == bool(parent_dir)
    offsets = COVERAGE_NODES if shared_plan else (LATER_NODES if start_step >= 2000 else NODES)
    expected_nodes = sorted({start_step+step for step in offsets if start_step+step <= steps} | {steps})
    paths_by_panel = dict.fromkeys(PATHS, PATHS['same']) if 'message_recipe' in meta else PATHS
    start, end = meta['start_hashes'], meta['end_hashes']
    assert start['oldpan'] == end['oldpan'] and start['decoder'] != end['decoder']
    assert (start['encoder'] == end['encoder']) == (meta['arm'] == 'j0')
    assert meta['test_images_read'] == 0 and meta['encoder_bn_stats_frozen'] and meta['decoder_bn_stats_frozen']
    assert sha256(directory / 'plan.json') == meta['plan_sha256']
    assert sha256(directory / 'checkpoints' / f'step-{steps:04d}.pt') == meta['final_checkpoint_sha256']
    assert all((directory / 'checkpoints' / f'step-{step:04d}.pt').is_file() for step in expected_nodes)
    if meta['arm'] == 'j1':
        probes = meta['gradient_probes']
        midpoint = start_step + (3000 if shared_plan else (1000 if start_step >= 2000 else 500))
        assert list(probes) == [str(step) for step in expected_nodes if step in (start_step, midpoint, steps)]
        for node in probes.values():
            assert set(node) == {'cage', 'trump'}
            assert all(np.isfinite(v[field]) and v[field] > 0 for v in node.values()
                       for field in ('encoder_grad_norm', 'image_grad_norm'))
    train = load_lines(directory / 'train.jsonl')
    assert [row['step'] for row in train] == list(range(start_step+1, steps+1))
    assert all(np.isfinite(row['combined_grad_norm']) and row['combined_grad_norm'] > 0 for row in train)
    assert all(np.isfinite(row[slot]['loss']) and row[slot]['max_linf'] <= 8/255+1e-7
               for row in train for slot in ('common','designated'))
    if 'message_recipe' in meta:
        verify_message_recipe(meta, train)
    nodes = load(directory / 'evaluation.json')
    assert [node['step'] for node in nodes] == expected_nodes
    evaluations = [validation(directory, node['step'], node['panels'], paths_by_panel) for node in nodes]
    parent_plan = None
    if parent_dir:
        parent = load(parent_dir / 'complete.json')
        assert parent['steps'] == start_step and meta['start_hashes'] == parent['end_hashes']
        assert meta['message_recipe'] == parent['message_recipe']
        assert meta['parent_checkpoint_sha256'] == parent['final_checkpoint_sha256'] == sha256(parent_dir / 'checkpoints' / f'step-{start_step:04d}.pt')
        assert meta['parent_plan_sha256'] == parent['plan_sha256'] == sha256(parent_dir / 'plan.json')
        assert sha256(directory / f'validation-{start_step:04d}.jsonl') == sha256(parent_dir / f'validation-{start_step:04d}.jsonl')
        parent_plan = load(parent_dir / 'plan.json')
    if shared_plan:
        verify_coverage_plan(directory, evaluations[0], meta, shared_plan)
    else:
        verify_plan(directory, evaluations[0], meta, parent_plan)
    for node in evaluations[1:]:
        assert all([(key(r), r['message']) for r in node[panel]['swap']] ==
                   [(key(r), r['message']) for r in evaluations[0][panel]['swap']] for panel in PATHS)
    tail = {slot: {field: float(np.mean([row[slot][field] for row in train[-100:]]))
                   for field in ('direct_bce', 'processed_bce', 'candidate_clipped', 'budget_saturated')}
            for slot in ('common','designated')}
    return meta, nodes, evaluations[0], evaluations[-1], tail


def permutation(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['sample_id']].append(row)
    n_covers, n_messages = len(groups), len(next(iter(groups.values())))
    assert all(len(group) == n_messages for group in groups.values())
    messages = np.array([[r['message'] for r in group] for group in groups.values()])
    predictions = np.array([[r['predicted_bits'] for r in group] for group in groups.values()])
    errors = (predictions[:, :, None, :] != messages[:, None, :, :]).sum(axis=3)
    draws = np.random.default_rng(20261109).random((10000, n_covers, n_messages)).argsort(axis=2)
    shuffled = errors[np.arange(n_covers)[None, :, None], np.arange(n_messages)[None, None, :], draws]
    null_exact = (shuffled == 0).sum(axis=(1, 2))
    null_bits = shuffled.sum(axis=(1, 2))
    exact = sum(r['bit_errors'] == 0 for r in rows)
    bits = sum(r['bit_errors'] for r in rows)
    return dict(observed_exact=exact, observed_ber=bits/(15*len(rows)),
                null_exact_95=np.quantile(null_exact, [.025, .975]).tolist(),
                null_exact_mean=float(null_exact.mean()), null_ber_mean=float(null_bits.mean()/(15*len(rows))),
                null_ber_95=(np.quantile(null_bits, [.025, .975])/(15*len(rows))).tolist(),
                exact_p_upper=float((1+(null_exact >= exact).sum())/10001),
                ber_p_lower=float((1+(null_bits <= bits).sum())/10001))


def contrasts(before, after):
    return {scope: paired(subset(before, scope), subset(after, scope)) for scope in SCOPES}


def compare(verified, comparison='arms'):
    if comparison == 'coverage':
        base_name = 'n16'
        controls = {'n64': ('n16',), 'n416': ('n16', 'n64')}
    elif comparison == 'recipes':
        base_name = 'mixed'
        controls = {'swap-weighted': ('mixed',), 'swap-equal': ('mixed', 'swap-weighted')}
    else:
        base_name = 'j0'
        controls = {'j1': ('j0', 'j2')}
    base_meta, _, base, _, _ = verified[base_name]
    resumed = base_meta.get('resume_from_step', 0)
    out = {}
    for arm, (meta, nodes, initial, final, tail) in verified.items():
        if resumed:
            assert meta['resume_from_step'] == resumed and meta['parent_plan_sha256'] == base_meta['parent_plan_sha256']
            assert all([(key(r),r['message']) for r in initial[panel]['swap']] ==
                       [(key(r),r['message']) for r in base[panel]['swap']] for panel in PATHS)
        else:
            assert initial == base and meta['start_hashes'] == base_meta['start_hashes'], 'Baseline rows differ across arms'
        plan_key = 'coverage_plan_sha256' if comparison == 'coverage' else 'plan_sha256'
        assert all(meta[k] == base_meta[k] for k in ('seed', 'steps', plan_key, 'checkpoint_sha256',
                                                     'manifest_sha256', 'design_sha256', 'script_sha256'))
        if comparison == 'coverage':
            assert meta['arm'] == 'j1' and arm == f"n{meta['carrier_count']}"
            assert meta['message_recipe'] == base_meta['message_recipe'] and meta['message_recipe'] in ('swap-weighted','swap-equal')
        if comparison == 'recipes':
            assert meta['arm'] == 'j1' and meta['message_recipe'] == arm
        out[arm] = dict(metadata=meta, trajectory=nodes, last_100_training_means=tail)
        for panel in PATHS:
            rows = final[panel]['swap']
            out[arm][panel] = dict(swap_vs_baseline=contrasts(initial[panel]['swap'], rows),
                                  swap_vs_clean=contrasts(final[panel]['clean_swap'], rows),
                                  permutation={scope: permutation(subset(rows, scope)) for scope in SCOPES})
            out[arm][panel].update({f'swap_vs_{control}': contrasts(verified[control][3][panel]['swap'], rows)
                                   for control in controls.get(arm, ())})
    if comparison in ('recipes', 'coverage'):
        return {comparison:out, 'primary_panel':'new'}
    j1 = out['j1']
    paths = j1['trajectory'][-1]['panels']['same']
    per_direction = {scope: paths['swap'][scope]['exact'] >= 13 and
                     j1['same']['permutation'][scope]['exact_p_upper'] < .05 and
                     j1['same']['swap_vs_clean'][scope]['wmr_delta'] > 0 for scope in ('cage', 'trump')}
    direct_delta = (paths['direct']['all']['exact'] -
                    j1['trajectory'][0]['panels']['same']['direct']['all']['exact'])/512
    gate = dict(per_direction=per_direction, direct_delta=direct_delta, direct_pass=direct_delta >= -.05,
                beats_controls=all(j1['same'][f'swap_vs_{arm}']['all']['wmr_delta'] > 0 for arm in ('j0','j2')))
    gate['passed'] = all(per_direction.values()) and gate['direct_pass'] and gate['beats_controls']
    return dict(arms=out, gate=gate)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--comparison', choices=('arms', 'recipes', 'coverage'), default='arms')
    parser.add_argument('--parent-root', type=Path, help='Root of the original runs for a continuation audit')
    args = parser.parse_args()
    groups = {'recipes':('mixed','swap-weighted','swap-equal'), 'coverage':('n16','n64','n416'),
              'arms':('j0','j1','j2')}[args.comparison]
    shared_plan = args.root / 'shared-plan.json' if args.comparison == 'coverage' else None
    verified = {arm: verify_run(args.root/f'{arm}-seed{args.seed}',
                                args.parent_root/f'{arm}-seed{args.seed}' if args.parent_root else None,
                                shared_plan) for arm in groups}
    result = compare(verified, args.comparison)
    name = 'analysis' if args.comparison == 'arms' else f'analysis-{args.comparison}'
    output = args.root / f'{name}-seed{args.seed}.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    for arm, entry in result[args.comparison].items():
        panels = entry['trajectory'][-1]['panels']
        print(arm, 'same', panels['same']['swap'], 'new', panels['new']['swap']['all'], flush=True)
    if 'gate' in result:
        print('gate', result['gate'], flush=True)
