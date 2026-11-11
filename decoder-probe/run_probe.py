"""Train a decoder against a frozen Author64 watermark and processing path."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

import diagnose_author64_output as protocol
from rgwm.cli.common import make_adapter, make_dataset
from rgwm.config import load_run_config
from rgwm.data.messages import split_message
from rgwm.evaluation.clean_quality import validate_clean_quality_report
from rgwm.evaluation.metrics import png_roundtrip
from rgwm.attacks.transforms import transform
from rgwm.legacy import file_hash
from rgwm.models.decoder import MessageDecoder


NODES = (0, 100, 250, 500, 750, 1000)
BASE_PATHS = ('direct', 'geometry_color_clamp', 'reconstruction', 'swap', 'clean_swap')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def tensor_digest(module):
    digest = hashlib.sha256()
    for name, tensor in module.state_dict().items():
        digest.update(name.encode())
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def metrics(values):
    count = len(values)
    return dict(n=count, exact=values.count(0), ber=sum(values) / (15 * count),
                error_buckets=dict(zero=values.count(0), one=values.count(1), two=values.count(2),
                                   three_or_more=sum(x >= 3 for x in values)))


def opposite(ids):
    return ['trump' if identity == 'cage' else 'cage' for identity in ids]


@torch.no_grad()
def protected_image(encoder, covers, bits):
    image = protocol.constrain(encoder(covers, bits), covers, protocol.EPSILON)
    require(torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1 and
            (image - covers).abs().max() <= protocol.EPSILON + 1e-7, 'Invalid watermark budget or RGB values')
    return image


@torch.no_grad()
def prepare_validation(dataset, design, encoder, adapter, device, extended):
    covers = torch.stack([dataset[i]['target'] for i in range(len(dataset))]).to(device)
    bits = torch.tensor(design['val_messages'], device=device, dtype=torch.float32)
    records = dataset.records
    ids = [row['identity_id'] for row in records]
    clean_swap = torch.cat([adapter.swap(None, covers[i:i+8], opposite(ids[i:i+8]), ids[i:i+8])
                            for i in range(0, len(covers), 8)])
    paths = list(BASE_PATHS)
    if extended:
        paths.extend(('png', 'jpeg50'))
    images = {name: [] for name in paths}
    labels, keys = [], []
    pairs = design['val_pairs']
    for start in range(0, len(pairs), 8):
        ii, jj = zip(*pairs[start:start+8])
        target_ids = [ids[i] for i in ii]
        cover, message = covers[list(ii)], bits[list(jj)]
        watermarked = protected_image(encoder, cover, message)
        normalized = adapter.restore_image(adapter.native_input(watermarked, target_ids)).clamp(0, 1)
        outputs = adapter.forward_pair(watermarked, opposite(target_ids), target_ids)
        batch = [watermarked, normalized, outputs.reconstruction, outputs.image, clean_swap[list(ii)]]
        if extended:
            batch += [png_roundtrip(watermarked), transform(watermarked, {'kind': 'jpeg', 'quality': 50},
                                                           differentiable=False)]
        for name, value in zip(paths, batch, strict=True):
            images[name].append(value)
        labels.extend(message.to(torch.int64).tolist())
        for i, j in zip(ii, jj, strict=True):
            row = records[i]
            keys.append(dict(sample_id=row['sample_id'], group_id=row['group_id'],
                             target_id=ids[i], message_index=j))
    return {name: torch.cat(batches) for name, batches in images.items()}, labels, keys


@torch.no_grad()
def evaluate(decoder, images, labels, keys, output):
    decoder.eval()
    summary = {}
    rows = []
    for name, batch in images.items():
        logits = torch.cat([decoder(part) for part in batch.split(8)])
        require(torch.isfinite(logits).all(), f'Nonfinite validation logits on {name}')
        pred = (logits >= 0).to(torch.int64).tolist()
        errors = [sum(x != y for x, y in zip(p, m, strict=True)) for p, m in zip(pred, labels, strict=True)]
        summary[name] = {'all': metrics(errors)}
        for identity in ('cage', 'trump'):
            summary[name][identity] = metrics([error for error, key in zip(errors, keys, strict=True)
                                               if key['target_id'] == identity])
        rows.extend(dict(key, path=name, message=label, predicted_bits=prediction, bit_errors=error)
                    for key, label, prediction, error in zip(keys, labels, pred, errors, strict=True))
    with output.open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row) + '\n')
    return summary


@torch.no_grad()
def training_images(encoder, adapter, covers, bits, ids, arm, step):
    image = protected_image(encoder, covers, bits)
    if arm == 'd1':
        return adapter.restore_image(adapter.native_input(image, ids)).clamp(0, 1)
    if arm == 'mixed' and step % 2 == 0:
        return image
    return adapter.swap(None, image, opposite(ids), ids)


@torch.no_grad()
def fit_diagnostic(decoder, encoder, adapter, covers, records, val, val_records, plan, val_messages, device):
    decoder.eval()
    indices = [i for batch in plan[:4] for i in batch['covers']]
    old = [m for batch in plan[:4] for m in batch['messages']]
    panels = (
        ('seen_cover_seen_pair', covers[indices], [records[i]['identity_id'] for i in indices], old),
        ('seen_cover_new_message', covers[indices], [records[i]['identity_id'] for i in indices],
         [val_messages[i % len(val_messages)] for i in range(len(old))]),
        ('new_cover_seen_message', val[:len(old)], [row['identity_id'] for row in val_records[:len(old)]], old),
    )
    result = {}
    for name, images, ids, labels in panels:
        messages = torch.tensor(labels, device=device, dtype=torch.float32)
        errors = []
        for i in range(0, len(images), 8):
            batch = images[i:i+8]
            bits = messages[i:i+8]
            outputs = training_images(encoder, adapter, batch, bits, ids[i:i+8], 'd2', 0)
            errors.extend(((decoder(outputs) >= 0) != bits).sum(1).tolist())
        result[name] = metrics(errors)
    return result


def run(args):
    protocol.SEED = args.seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    prior = read_json(args.checkpoint.parent / 'complete.json')
    historical = read_json(args.historical)
    require(file_hash(args.checkpoint) == prior['checkpoint_sha256'] == historical['checkpoint_sha256'],
            'Checkpoint hash mismatch')
    manifest_hash = file_hash(cfg['assets']['manifest'])
    require(manifest_hash == design['manifest_sha256'] == prior['manifest_sha256'] == historical['manifest_sha256'],
            'Manifest hash mismatch')
    require(file_hash(args.design) == historical['design_sha256'], 'Expected the same design as experiment A')
    require(cfg['model']['message_length'] == 15 and cfg['budget']['epsilon'] == protocol.EPSILON,
            'Expected 15 bits and the frozen watermark budget')
    train = make_dataset(cfg, split='train')
    val = make_dataset(cfg, split='val')
    require((len(train), len(val)) == (416, 139) and not train.augment and not val.augment,
            'Expected the frozen train/val split without augmentation')
    require([r['sample_id'] for r in val.records] == design['val_ids'], 'Unexpected validation order')
    pairs = [tuple(p) for p in design['val_pairs']]
    require(len(pairs) == len(set(pairs)) == 556 and Counter(i for i, _ in pairs) == dict.fromkeys(range(139), 4),
            'Expected four distinct validation messages per cover')
    require(all(split_message(m) == 'val' for m in design['val_messages']), 'Unexpected validation messages')
    expected_order = read_json(args.design.parent / 'complete.json')['train_ids']
    lookup = {r['sample_id']: i for i, r in enumerate(train.records)}
    require(len(expected_order) == len(set(expected_order)) == 416 and set(expected_order) == set(lookup),
            'Unexpected training carrier manifest')
    order = [lookup[i] for i in expected_order]
    records = [train.records[i] for i in order]
    covers = torch.stack([train[i]['target'] for i in order]).to(device)
    bank = protocol.dynamic_bank(design['train_messages'], 1000)
    require(len(bank) == len(set(map(tuple, bank))) == 16000 and all(split_message(m) == 'train' for m in bank),
            'Expected 16000 distinct training messages')
    cover_stream = protocol.full_cover_stream(records)
    plan = [dict(covers=next(cover_stream), messages=bank[i:i+16]) for i in range(0, len(bank), 16)]
    require({i for batch in plan for i in batch['covers']} == set(range(416)), 'Incomplete training cover exposure')
    require(all(Counter(records[i]['identity_id'] for i in batch['covers']) == {'cage': 8, 'trump': 8}
                and all(sum(col) == 8 for col in zip(*batch['messages'])) for batch in plan), 'Unbalanced batch')
    if args.small:
        selected = next(protocol.full_cover_stream(records))
        # Four distinct messages per cover, for a fixed training-only fit check.
        plan = [dict(covers=selected, messages=bank[i:i+16]) for i in range(0, 64, 16)] * 250
    adapter = make_adapter(cfg, device).eval().requires_grad_(False)
    validate_clean_quality_report(cfg, adapter)
    state = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    encoder = protocol.OutputEncoder('residual').to(device).eval().requires_grad_(False)
    encoder.load_state_dict(state['encoder'], strict=True)
    decoder = MessageDecoder(15).to(device)
    decoder.load_state_dict(state['decoder'], strict=True)
    del state
    frozen = dict(encoder=tensor_digest(encoder), oldpan=tensor_digest(adapter))
    initial_decoder = tensor_digest(decoder)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'training_plan.json', dict(ordered_train_ids=expected_order, batches=plan))
    metadata = dict(arm=args.arm, seed=args.seed, gpu=torch.cuda.get_device_name(device), device=str(device),
                    lr=args.lr, optimizer='Adam', weight_decay=1e-5, trainable='decoder_only',
                    bn_running_stats_frozen=not args.adapt_bn, small=args.small, steps=1000, batch_size=16,
                    micro_batch=8, checkpoint_sha256=file_hash(args.checkpoint), manifest_sha256=manifest_hash,
                    design_sha256=file_hash(args.design), plan_sha256=file_hash(args.output / 'training_plan.json'),
                    adapter_sha256=adapter.checkpoint_hash, script_sha256=file_hash(__file__),
                    validation_images=139, validation_pairs=556, test_images_read=0,
                    torch_version=torch.__version__, frozen_before=frozen, decoder_initial=initial_decoder)
    write_json(args.output / 'config.json', metadata)
    validation, labels, keys = prepare_validation(val, design, encoder, adapter, device, args.arm == 'mixed')
    require(Counter(k['target_id'] for k in keys) == {'cage': 256, 'trump': 300}, 'Wrong direction counts')
    optimizer = torch.optim.Adam(decoder.parameters(), lr=args.lr, weight_decay=1e-5)
    bn_layers = [layer for layer in decoder.modules() if isinstance(layer, nn.BatchNorm2d)]
    bn_initial = [(layer.running_mean.clone(), layer.running_var.clone(), layer.num_batches_tracked.clone())
                  for layer in bn_layers]
    checkpoints = args.output / 'checkpoints'
    checkpoints.mkdir()
    evaluations = []

    def save_node(step):
        results = evaluate(decoder, validation, labels, keys, args.output / f'validation-{step:04d}.jsonl')
        evaluations.append(dict(step=step, paths=results))
        write_json(args.output / 'evaluation.json', evaluations)
        torch.save(dict(step=step, data_cursor=step*16, decoder=decoder.state_dict(),
                        optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device), python_rng=random.getstate(), metadata=metadata),
                   checkpoints / f'step-{step:04d}.pt')
        print(json.dumps(dict(step=step, swap=results['swap']['all'], direct=results['direct']['all'])), flush=True)
        return results

    results = save_node(0)
    for name in BASE_PATHS[:-1]:
        for scope in ('all', 'cage', 'trump'):
            expected = historical['by_path'][name][scope]
            actual = results[name][scope]
            require(actual['exact'] == expected['exact'] and abs(actual['ber'] - expected['ber']) < 1e-12,
                    f'D0 baseline replay mismatch on {name}/{scope}')

    torch.manual_seed(args.seed)
    with (args.output / 'train.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
        for step, batch in enumerate(plan, start=1):
            decoder.train()
            if not args.adapt_bn:
                for layer in bn_layers:
                    layer.eval()
            optimizer.zero_grad(set_to_none=True)
            loss_total = 0.
            errors = []
            for start in (0, 8):
                ii = batch['covers'][start:start+8]
                bits = torch.tensor(batch['messages'][start:start+8], device=device, dtype=torch.float32)
                ids = [records[i]['identity_id'] for i in ii]
                images = training_images(encoder, adapter, covers[ii], bits, ids, args.arm, step)
                logits = decoder(images)
                loss = F.binary_cross_entropy_with_logits(logits, bits)
                require(torch.isfinite(loss), f'Nonfinite loss at step {step}')
                (loss / 2).backward()
                loss_total += float(loss.detach()) / 2
                errors.extend(((logits.detach() >= 0) != bits).sum(1).tolist())
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), float('inf'), error_if_nonfinite=True)
            optimizer.step()
            stream.write(json.dumps(dict(step=step, bce=loss_total, **metrics(errors)), allow_nan=False) + '\n')
            if step in NODES:
                save_node(step)
            elif step % 50 == 0:
                print(f'completed {step} updates, latest training BCE {loss_total:.5f}', flush=True)
    require(len(evaluations) == len(NODES), 'Missing evaluation node')
    final_frozen = dict(encoder=tensor_digest(encoder), oldpan=tensor_digest(adapter))
    require(final_frozen == frozen and not any(p.grad is not None for module in (encoder, adapter)
                                                for p in module.parameters()), 'A frozen model changed')
    if not args.adapt_bn:
        require(all(torch.equal(actual, expected) for layer, state in zip(bn_layers, bn_initial, strict=True)
                    for actual, expected in zip((layer.running_mean, layer.running_var, layer.num_batches_tracked),
                                                state, strict=True)), 'Frozen BN statistics changed')
    diag = fit_diagnostic(decoder, encoder, adapter, covers, records,
                          torch.stack([val[i]['target'] for i in range(64)]).to(device), val.records,
                          plan, design['val_messages'], device)
    write_json(args.output / 'complete.json', dict(metadata, frozen_after=final_frozen,
               decoder_final=tensor_digest(decoder), fit_diagnostic=diag,
               final_checkpoint_sha256=file_hash(checkpoints / 'step-1000.pt')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assets', 'design', 'checkpoint', 'historical', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--arm', choices=('d1', 'd2', 'mixed'), required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--adapt-bn', action='store_true')
    parser.add_argument('--small', action='store_true')
    run(parser.parse_args())


if __name__ == '__main__':
    main()
