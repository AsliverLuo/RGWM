"""Replay frozen Author64 validation pairs through six diagnostic paths."""

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch

from diagnose_author64_output import EPSILON, MICRO, OutputEncoder, constrain
from rgwm.adapters.trump_cage import sample_grid
from rgwm.cli.common import make_adapter, make_dataset
from rgwm.config import load_run_config
from rgwm.evaluation.clean_quality import validate_clean_quality_report
from rgwm.legacy import file_hash
from rgwm.models.decoder import MessageDecoder


PATHS = ('direct', 'geometry_only', 'geometry_color', 'geometry_color_clamp',
         'reconstruction', 'swap')
COMPARISONS = (('direct', 'geometry_only'), ('geometry_only', 'geometry_color'),
               ('geometry_color', 'geometry_color_clamp'),
               ('geometry_color', 'reconstruction'), ('reconstruction', 'swap'))
IDENTITIES = ('cage', 'trump')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def path_metrics(errors):
    histogram = np.bincount(errors, minlength=16)
    return dict(n=len(errors), exact=int(histogram[0]), ber=float(errors.mean() / 15),
                wmr=float(histogram[0] / len(errors)), error_histogram=histogram.tolist())


def paired_metrics(errors, indices, records, scope):
    groups = {}
    for position, index in enumerate(indices):
        groups.setdefault(records[index]['group_id'], []).append(position)
    groups = list(groups.values())
    sizes = np.array([len(group) for group in groups])
    rng = np.random.default_rng(20261109)
    draws = rng.integers(len(groups), size=(10000, len(groups)))
    denominators = sizes[draws].sum(axis=1)
    result = {}
    for before, after in COMPARISONS:
        x, y = errors[before][indices], errors[after][indices]
        deltas = np.column_stack(((y - x) / 15, (y == 0).astype(int) - (x == 0)))
        grouped = np.stack([deltas[group].sum(axis=0) for group in groups])
        means = grouped[draws].sum(axis=1) / denominators[:, None]
        intervals = np.quantile(means, [.025, .975], axis=0).T
        result[f'{after}_minus_{before}'] = dict(
            scope=scope, n=len(indices), groups=len(groups),
            ber_delta=float(deltas[:, 0].mean()), ber_ci95=intervals[0].tolist(),
            wmr_delta=float(deltas[:, 1].mean()), wmr_ci95=intervals[1].tolist(),
            exact_gained=int(((x != 0) & (y == 0)).sum()),
            exact_lost=int(((x == 0) & (y != 0)).sum()))
    return result


@torch.inference_mode()
def run(args):
    design = load_json(args.design)
    historical = load_json(args.historical)['results'][args.checkpoint.parent.name]
    checkpoint_hash = file_hash(args.checkpoint)
    checkpoint_info = load_json(args.checkpoint.parent / 'complete.json')
    require(checkpoint_hash == historical['checkpoint_sha256'] == checkpoint_info['checkpoint_sha256'],
            'Checkpoint hash differs from completion and historical records')
    cfg = load_run_config(args.config, args.assets.resolve())
    manifest_hash = file_hash(cfg['assets']['manifest'])
    require(manifest_hash == design['manifest_sha256'] == checkpoint_info['manifest_sha256'],
            'Manifest hash differs from the frozen design or checkpoint')
    require(cfg['model']['message_length'] == 15 and cfg['budget']['epsilon'] == EPSILON,
            'Expected 15 bits and 8/255 budget')
    dataset = make_dataset(cfg, split='val')
    require(len(dataset) == 139 and not dataset.augment, 'Expected 139 unaugmented val covers')
    require([record['sample_id'] for record in dataset.records] == design['val_ids'],
            'Validation cover order differs from the frozen design')
    pairs = [tuple(pair) for pair in design['val_pairs']]
    require(len(pairs) == len(set(pairs)) == 556 and
            Counter(i for i, _ in pairs) == dict.fromkeys(range(139), 4),
            'Expected four distinct messages per validation cover')
    records = [dataset.records[i] for i, _ in pairs]
    require(Counter(row['identity_id'] for row in records) == {'cage': 256, 'trump': 300},
            'Unexpected direction counts')
    device = torch.device(args.device)
    covers = torch.stack([dataset[i]['target'] for i in range(len(dataset))]).to(device)
    val_messages = torch.tensor(design['val_messages'], dtype=torch.float32, device=device)
    adapter = make_adapter(cfg, device).eval()
    validate_clean_quality_report(cfg, adapter)
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    encoder = OutputEncoder('residual').to(device).eval()
    decoder = MessageDecoder(15).to(device).eval()
    encoder.load_state_dict(state['encoder'], strict=True)
    decoder.load_state_dict(state['decoder'], strict=True)
    del state
    down = torch.arange(64, device=device) * 2.5
    up = torch.arange(160, device=device) * .4
    errors = {path: [] for path in PATHS}
    cage_max_delta = 0.
    max_linf = 0.
    range_stats = {key: Counter() for key in ('all', *IDENTITIES)}
    rows = []
    for start in range(0, len(pairs), MICRO):
        cover_ids, msg_ids = zip(*pairs[start:start + MICRO])
        batch_records = records[start:start + MICRO]
        target_ids = [row['identity_id'] for row in batch_records]
        source_ids = ['trump' if identity == 'cage' else 'cage' for identity in target_ids]
        cover, message = covers[list(cover_ids)], val_messages[list(msg_ids)]
        protected = constrain(encoder(cover, message), cover, EPSILON)
        actual_linf = (protected - cover).abs().flatten(1).amax(1)
        max_linf = max(max_linf, float(actual_linf.max()))
        require(bool(torch.isfinite(protected).all()) and float(protected.min()) >= 0 and
                float(protected.max()) <= 1 and max_linf <= EPSILON + 1e-7, 'Watermark contract failed')
        geometry = sample_grid(sample_grid(protected, down), up)
        color = adapter.restore_image(adapter.native_input(protected, target_ids))
        outputs = adapter.forward_pair(protected, source_ids, target_ids)
        candidates = (protected, geometry, color, color.clamp(0, 1),
                      outputs.reconstruction, outputs.image)
        for i, identity in enumerate(target_ids):
            if identity == 'cage':
                cage_max_delta = max(cage_max_delta, float((geometry[i] - color[i]).abs().max()))
        for path, images in zip(PATHS, candidates, strict=True):
            logits = decoder(images)
            require(bool(torch.isfinite(logits).all()), f'Nonfinite logits for {path}')
            predictions = (logits >= 0).to(torch.int32)
            bit_errors = (predictions != message).sum(1).tolist()
            errors[path].extend(bit_errors)
            image_linf = actual_linf.tolist()
            below = (images < 0).sum((1, 2, 3)).tolist()
            above = (images > 1).sum((1, 2, 3)).tolist()
            require(path == 'geometry_color' or not (any(below) or any(above)),
                    f'Unexpected range violation in {path}')
            for i, prediction in enumerate(predictions.tolist()):
                record = batch_records[i]
                target, source = target_ids[i], source_ids[i]
                output_identity = source if path == 'swap' else target if path == 'reconstruction' else None
                rows.append(dict(path=path, sample_id=record['sample_id'], group_id=record['group_id'],
                                 message_index=msg_ids[i], message=design['val_messages'][msg_ids[i]],
                                 target_id=target, planned_swap_id=source, output_branch_identity=output_identity,
                                 predicted_bits=prediction, bit_errors=bit_errors[i], watermark_linf=image_linf[i],
                                 below_zero=below[i], above_one=above[i], n_channel_values=images[i].numel()))
                if path == 'geometry_color':
                    for scope in ('all', target):
                        range_stats[scope].update(dict(n_channel_values=images[i].numel(),
                                                       below_zero=below[i], above_one=above[i],
                                                       pairs_outside=int(bool(below[i] or above[i]))))
    require(cage_max_delta <= 1e-6, 'Cage geometry and color path differ beyond tolerance')
    require(len(rows) == 556 * len(PATHS), 'Incorrect output row count')
    errors = {path: np.array(values) for path, values in errors.items()}
    scopes = {'all': np.arange(len(records))}
    scopes.update({identity: np.array([i for i, r in enumerate(records) if r['identity_id'] == identity])
                   for identity in IDENTITIES})
    by_path = {path: {scope: path_metrics(values[indices]) for scope, indices in scopes.items()}
               for path, values in errors.items()}
    replay = []
    for path, scopes_expected in historical['by_path'].items():
        for scope, expected in scopes_expected.items():
            actual = by_path[path][scope]
            match = all(actual[key] == expected[key] for key in ('n', 'exact')) and abs(actual['ber'] - expected['ber']) < 1e-12
            replay.append(dict(path=path, scope=scope, expected=expected,
                               actual={key: actual[key] for key in ('n', 'exact', 'ber')}, match=match))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    summary = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=checkpoint_hash,
                   manifest_sha256=manifest_hash, design_sha256=file_hash(args.design),
                   historical_sha256=file_hash(args.historical), adapter_weights_sha256=adapter.checkpoint_hash,
                   adapter_preprocess=adapter.preprocess_spec, assets_sha256=file_hash(args.assets),
                   script_sha256=file_hash(__file__), device=str(device), torch_version=torch.__version__,
                   gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                   validation_images=len(dataset), validation_pairs=len(pairs), prediction_rows=len(rows),
                   test_images_read=0, epsilon=EPSILON, max_watermark_linf=max_linf,
                   cage_geometry_color_max_abs_difference=cage_max_delta,
                   geometry_color_range=range_stats, by_path=by_path, historical_replay=replay,
                   paired={scope: paired_metrics(errors, indices, records, scope)
                           for scope, indices in scopes.items()})
    write_json(args.output / 'summary.json', summary)
    print(json.dumps({path: metrics['all'] for path, metrics in by_path.items()}), flush=True)
    require(all(row['match'] for row in replay), f'Historical replay mismatch, inspect {args.output / "summary.json"}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assets', 'design', 'historical', 'checkpoint', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--device', default='cuda:0')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
