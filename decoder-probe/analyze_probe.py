"""Verify probe outputs and test whether predictions follow embedded messages."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


NODES = [0, 100, 250, 500, 750, 1000]
SCOPES = ('all', 'cage', 'trump')


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def load_lines(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def key(row):
    return row['sample_id'], row['message_index']


def subset(rows, scope):
    return [row for row in rows if scope == 'all' or row['target_id'] == scope]


def by_path(directory, step, expected):
    paths = {name: {} for name in expected}
    for row in load_lines(directory / f'validation-{step:04d}.jsonl'):
        path = paths[row['path']]
        assert key(row) not in path
        assert len(row['message']) == len(row['predicted_bits']) == 15
        assert set(row['message'] + row['predicted_bits']) <= {0, 1}
        assert sum(a != b for a, b in zip(row['message'], row['predicted_bits'], strict=True)) == row['bit_errors']
        path[key(row)] = row
    order = sorted(paths['swap'])
    assert len(order) == 556 and len({sample for sample, _ in order}) == 139
    assert set(Counter(sample for sample, _ in order).values()) == {4}
    for name, path in paths.items():
        assert sorted(path) == order
        rows = [path[k] for k in order]
        assert all(row['message'] == paths['swap'][key(row)]['message'] for row in rows)
        for scope in SCOPES:
            errors = [row['bit_errors'] for row in subset(rows, scope)]
            actual = expected[name][scope]
            assert len(errors) == actual['n'] == {'all': 556, 'cage': 256, 'trump': 300}[scope]
            assert actual['exact'] == errors.count(0)
            assert abs(actual['ber'] - sum(errors) / (15 * len(errors))) < 1e-12
            assert actual['error_buckets'] == dict(zero=errors.count(0), one=errors.count(1), two=errors.count(2),
                                                    three_or_more=sum(x >= 3 for x in errors))
    return {name: [path[k] for k in order] for name, path in paths.items()}


def verify_run(directory):
    metadata = load(directory / 'complete.json')
    assert metadata['frozen_before'] == metadata['frozen_after']
    assert metadata['decoder_initial'] != metadata['decoder_final']
    assert metadata['test_images_read'] == 0
    assert sha256(directory / 'training_plan.json') == metadata['plan_sha256']
    assert sha256(directory / 'checkpoints' / 'step-1000.pt') == metadata['final_checkpoint_sha256']
    assert all((directory / 'checkpoints' / f'step-{step:04d}.pt').is_file() for step in NODES)
    train = load_lines(directory / 'train.jsonl')
    assert [row['step'] for row in train] == list(range(1, 1001))
    assert all(np.isfinite(row['bce']) and row['n'] == 16 for row in train)
    evaluations = load(directory / 'evaluation.json')
    assert [node['step'] for node in evaluations] == NODES
    files = [by_path(directory, node['step'], node['paths']) for node in evaluations]
    assert all([row['message'] for row in node['swap']] == [row['message'] for row in files[0]['swap']]
               for node in files)
    return metadata, evaluations, files[0], files[-1], {
        'first_100_bce': float(np.mean([row['bce'] for row in train[:100]])),
        'last_100_bce': float(np.mean([row['bce'] for row in train[-100:]])),
        'last_100_exact': sum(row['exact'] for row in train[-100:]),
        'last_100_pairs': 1600,
        'last_100_ber': float(np.mean([row['ber'] for row in train[-100:]])),
    }


def paired(before, after):
    assert [(key(r), r['message']) for r in before] == [(key(r), r['message']) for r in after]
    a = np.array([r['bit_errors'] for r in before])
    b = np.array([r['bit_errors'] for r in after])
    deltas = np.column_stack(((b - a) / 15, (b == 0).astype(int) - (a == 0)))
    groups = {}
    for i, row in enumerate(after):
        groups.setdefault(row['group_id'], []).append(i)
    groups = list(groups.values())
    weights = np.array([len(g) for g in groups])
    sums = np.stack([deltas[g].sum(axis=0) for g in groups])
    draws = np.random.default_rng(20261109).integers(len(groups), size=(10000, len(groups)))
    bootstrap = sums[draws].sum(axis=1) / weights[draws].sum(axis=1)[:, None]
    bounds = np.quantile(bootstrap, [.025, .975], axis=0).T
    return dict(n=len(a), groups=len(groups), ber_delta=float(deltas[:, 0].mean()), ber_ci95=bounds[0].tolist(),
                wmr_delta=float(deltas[:, 1].mean()), wmr_ci95=bounds[1].tolist(),
                exact_gained=int(((a != 0) & (b == 0)).sum()), exact_lost=int(((a == 0) & (b != 0)).sum()))


def within_cover_permutation(rows):
    by_cover = {}
    for row in rows:
        by_cover.setdefault(row['sample_id'], []).append(row)
    assert all(len(group) == 4 for group in by_cover.values())
    messages = np.array([[r['message'] for r in group] for group in by_cover.values()])
    predictions = np.array([[r['predicted_bits'] for r in group] for group in by_cover.values()])
    errors = (predictions[:, :, None, :] != messages[:, None, :, :]).sum(axis=3)
    draws = np.random.default_rng(20261109).random((10000, len(by_cover), 4)).argsort(axis=2)
    shuffled = errors[np.arange(len(by_cover))[None, :, None], np.arange(4)[None, None, :], draws]
    shuffled_bits = shuffled.sum(axis=(1, 2))
    shuffled_exact = (shuffled == 0).sum(axis=(1, 2))
    observed_bits = sum(r['bit_errors'] for r in rows)
    observed_exact = sum(r['bit_errors'] == 0 for r in rows)
    return dict(permutations=len(draws), n=len(rows), observed_exact=observed_exact,
                observed_ber=observed_bits / (15 * len(rows)),
                null_exact_mean=float(shuffled_exact.mean()),
                null_exact_95=np.quantile(shuffled_exact, [.025, .975]).tolist(),
                null_ber_mean=float(shuffled_bits.mean() / (15 * len(rows))),
                null_ber_95=(np.quantile(shuffled_bits, [.025, .975]) / (15 * len(rows))).tolist(),
                exact_p_upper=float((1 + (shuffled_exact >= observed_exact).sum()) / (1 + len(draws))),
                ber_p_lower=float((1 + (shuffled_bits <= observed_bits).sum()) / (1 + len(draws))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verified = {path.name: verify_run(path) for path in args.run}
    baseline = next(iter(verified.values()))[2]
    for metadata, _, initial, _, _ in verified.values():
        assert all(initial[path] == baseline[path] for path in baseline if path in initial)
        assert metadata['frozen_before'] == next(iter(verified.values()))[0]['frozen_before']
    for name, (metadata, _, _, _, _) in verified.items():
        if name.startswith('d2-seed') and 'd1-' + name[3:] in verified:
            other = verified['d1-' + name[3:]][0]
            assert all(metadata[field] == other[field] for field in ('seed', 'lr', 'plan_sha256', 'decoder_initial',
                                                                    'bn_running_stats_frozen', 'script_sha256'))
    result = {}
    for name, (metadata, evaluations, _, final, train) in verified.items():
        same_seed = verified.get('d1-seed' + str(metadata['seed']))
        result[name] = dict(metadata=metadata, train=train, trajectory=evaluations,
                            swap_vs_initial={scope: paired(subset(baseline['swap'], scope), subset(final['swap'], scope))
                                             for scope in SCOPES},
                            swap_vs_clean={scope: paired(subset(final['clean_swap'], scope), subset(final['swap'], scope))
                                           for scope in SCOPES},
                            permutation={scope: within_cover_permutation(subset(final['swap'], scope)) for scope in SCOPES})
        if same_seed is not None and metadata['arm'] != 'd1':
            result[name]['swap_vs_d1'] = {scope: paired(subset(same_seed[3]['swap'], scope), subset(final['swap'], scope))
                                         for scope in SCOPES}
        if metadata['arm'] == 'd2':
            result[name]['gate_met'] = all(result[name]['permutation'][scope]['observed_exact'] >= minimum and
                                           result[name]['permutation'][scope]['exact_p_upper'] < .05
                                           for scope, minimum in (('cage', 13), ('trump', 15)))
        print(name, json.dumps({'swap': evaluations[-1]['paths']['swap'],
                                'permutation': result[name]['permutation']['all'],
                                'gate_met': result[name].get('gate_met')}), flush=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
