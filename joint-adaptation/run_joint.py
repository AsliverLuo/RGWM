"""Matched small-carrier joint adaptation with a frozen Oldpan victim."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random

import torch
from torch import nn
from torch.nn import functional as F

import diagnose_author64_output as protocol
from run_probe import metrics, opposite, read_json, require, tensor_digest, write_json
from train_author64_distortion import training_transform
from rgwm.attacks.transforms import transform
from rgwm.cli.common import make_adapter, make_dataset
from rgwm.config import load_run_config
from rgwm.data.messages import split_message
from rgwm.evaluation.clean_quality import validate_clean_quality_report
from rgwm.evaluation.metrics import png_roundtrip
from rgwm.legacy import file_hash
from rgwm.models.decoder import MessageDecoder
from rgwm.training.image_losses import image_structure_loss


NODES = (0, 100, 250, 500, 750, 1000)
LATER_NODES = tuple(range(0, 2001, 250))
EPSILON = 8 / 255
IDENTITIES = ('cage', 'trump')
EVALUATION_PATHS = ('swap', 'clean_swap', 'direct', 'png', 'jpeg50')
MESSAGE_WEIGHTS = {'mixed': (.75, .25), 'swap-weighted': (.75, .25), 'swap-equal': (.5, .5)}


def continuation_bank(anchor, consumed, count, rng):
    anchor = set(map(tuple, anchor))
    fresh, recycled = [], []
    for value in range(1 << 14):
        bits = tuple(map(int, f'{value:015b}'))
        opposite_bits = tuple(1 - bit for bit in bits)
        if bits in anchor or opposite_bits in anchor or split_message(bits) != 'train' or split_message(opposite_bits) != 'train':
            continue
        require((bits in consumed) == (opposite_bits in consumed), 'Historical message pair was split')
        (recycled if bits in consumed else fresh).append((bits, opposite_bits))
    rng.shuffle(fresh)
    rng.shuffle(recycled)
    pairs = (fresh + recycled)[:count//2]
    require(2*len(pairs) == count, 'Insufficient unique messages in continuation segment')
    return [list(message) for pair in pairs for message in pair]


def balanced_reuse_bank(consumed, count, rng):
    """Fill the least-used complementary pairs before starting another pass."""
    frequency = {}
    for bits, uses in consumed.items():
        opposite_bits = tuple(1 - bit for bit in bits)
        require(consumed[opposite_bits] == uses and split_message(bits) == 'train',
                'Historical training message counts are invalid')
        if bits[0] == 0:
            frequency[bits, opposite_bits] = uses
    require(frequency and max(frequency.values()) - min(frequency.values()) <= 1,
            'Historical message passes are unbalanced')
    remaining = count // 2
    bank = []
    while remaining:
        level = min(frequency.values())
        pool = [pair for pair, uses in frequency.items() if uses == level]
        rng.shuffle(pool)
        selected = pool[:remaining]
        for pair in selected:
            frequency[pair] += 1
            bank.extend(list(bits) for bits in pair)
        remaining -= len(selected)
    return bank


def prepare_plan(train, design, seed, steps, parent_plan=None):
    protocol.SEED = 0
    selected = protocol.chosen_indices(train)
    ids = [train.records[i]['identity_id'] for i in selected]
    require(Counter(ids) == {'cage': 8, 'trump': 8}, 'Expected 8 carriers per identity')
    protocol.SEED = seed
    start = len(parent_plan['batches']) if parent_plan else 0
    tail_rng = random.Random(20261111 + seed + (start if start >= 2000 else 0))
    if start >= 2000:
        bank = [m for batch in parent_plan['batches'] for micro in batch['micros'] for m in micro['message']]
        require(len(bank) == 16*start, 'Historical plan message count mismatch')
        bank.extend(balanced_reuse_bank(Counter(map(tuple, bank)), 16*(steps-start), tail_rng))
    else:
        bank = protocol.dynamic_bank(design['train_messages'], start or steps)
        require(len(bank) == len(set(map(tuple, bank))) == 16*(start or steps) and
                all(split_message(m) == 'train' for m in bank), 'Expected distinct train messages')
        if start:
            bank.extend(continuation_bank(design['train_messages'], set(map(tuple, bank)), 16*(steps-start), tail_rng))
    same_messages = protocol.balanced_messages('val', 32, 20261109)
    require(len(set(same_messages)) == 32 and all(split_message(m) == 'val' for m in same_messages),
            'Unexpected same-carrier evaluation messages')
    rng = random.Random(20261109 + seed)
    augment = random.Random(20261110 + seed)
    order = {identity: [i for i, label in enumerate(ids) if label == identity] for identity in IDENTITIES}
    batches = []
    exposures = Counter()
    historical_pairs = set()
    for step in range(steps):
        if step % 2 == 0:
            for positions in order.values():
                rng.shuffle(positions)
        slot = (step % 2) * 4
        designated = [i for identity in IDENTITIES for i in order[identity][slot:slot+4]]
        common = [i for i in range(16) if i not in designated]
        rng.shuffle(designated)
        rng.shuffle(common)
        # Alternate which microbatch receives the designated path.
        micros = [dict(slot='designated', indices=designated), dict(slot='common', indices=common)]
        if step % 2:
            micros.reverse()
        messages = bank[16*step:16*(step+1)]
        indices = [i for micro in micros for i in micro['indices']]
        if start and step >= start:
            for _ in range(1000):
                tail_rng.shuffle(messages)
                if all((i, tuple(m)) not in historical_pairs for i, m in zip(indices, messages, strict=True)):
                    break
            else:
                raise ValueError('Could not assign recycled messages to different carriers')
        if start:
            historical_pairs.update((i, tuple(m)) for i, m in zip(indices, messages, strict=True))
        for j, micro in enumerate(micros):
            micro['message'] = messages[8*j:8*(j+1)]
            require(Counter(ids[i] for i in micro['indices']) == {'cage': 4, 'trump': 4}, 'Imbalanced slot')
            exposures.update((micro['slot'], i) for i in micro['indices'])
        batches.append(dict(micros=micros, transform=training_transform(augment)))
    require(all(abs(2*exposures[slot, i] - steps) <= 1 for slot in ('common', 'designated')
                for i in range(16)), 'Unequal exposure to designated path')
    plan = dict(train_ids=[train.records[i]['sample_id'] for i in selected], same_messages=same_messages, batches=batches)
    if start:
        prefix = dict(plan, batches=batches[:start])
        require(json.loads(json.dumps(prefix)) == parent_plan, 'Continuation plan changes the historical prefix')
    return selected, plan


def watermarked(encoder, cover, message, priors=None):
    if priors is not None:
        result = encoder(cover, message, priors)
        with torch.no_grad():
            fused = result.adv_gate * result.adv_residual + result.msg_gate * result.msg_residual
            candidate = cover + fused
        return result.image, candidate
    candidate = encoder(cover, message)
    image = protocol.constrain(candidate, cover, EPSILON)
    return image, candidate


@torch.no_grad()
def quality(cover, image, candidate):
    delta = (image - cover).abs().flatten(1)
    cap = delta.amax(dim=1)
    require(torch.isfinite(image).all() and image.min() >= 0 and image.max() <= 1 and
            cap.max() <= EPSILON + 1e-7, 'RGB or watermark budget violation')
    clipping = ((candidate < (cover-EPSILON).clamp_min(0)) |
                (candidate > (cover+EPSILON).clamp_max(1))).flatten(1).float().mean(dim=1)
    at_budget = (delta >= EPSILON-1e-7).float().mean(dim=1)
    mse = delta.square().mean(dim=1)
    return list(zip(cap.tolist(), clipping.tolist(), at_budget.tolist(), mse.tolist(), strict=True))


def freeze_bn(module):
    for layer in module.modules():
        if isinstance(layer, nn.modules.batchnorm._BatchNorm):
            layer.eval()


def bn_digest(module):
    # Track buffers separately from affine weights.
    return [(name, buffer.detach().cpu().clone()) for name, buffer in module.named_buffers()
            if name.endswith(('running_mean', 'running_var', 'num_batches_tracked'))]


def gradient_probe(encoder, decoder, adapter, covers, ids, plan, device):
    encoder.eval()
    decoder.eval()
    parameters = [p for p in encoder.parameters() if p.requires_grad]
    results = {}
    for identity in IDENTITIES:
        indices = [i for i, label in enumerate(ids) if label == identity][:4]
        labels = torch.tensor(plan['batches'][0]['micros'][0]['message'][:4], device=device, dtype=torch.float32)
        image, _ = watermarked(encoder, covers[indices], labels)
        out = adapter.swap(None, image, opposite([identity]*4), [identity]*4)
        loss = F.binary_cross_entropy_with_logits(decoder(out), labels)
        image_gradient, *grads = torch.autograd.grad(loss, [image, *parameters], allow_unused=True)
        encoder_norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in grads if g is not None))
        image_norm = float(image_gradient.detach().norm())
        require(math.isfinite(encoder_norm) and encoder_norm > 0 and math.isfinite(image_norm) and image_norm > 0,
                f'Swap BCE gradient missing for {identity}')
        results[identity] = dict(encoder_grad_norm=encoder_norm, image_grad_norm=image_norm, bce=float(loss.detach()))
    return results


@torch.no_grad()
def clean_outputs(adapter, covers, ids):
    return torch.cat([adapter.swap(None, covers[i:i+8], opposite(ids[i:i+8]), ids[i:i+8])
                      for i in range(0, len(covers), 8)])


@torch.no_grad()
def evaluate(encoder, decoder, adapter, panels, output, *, observe_swap=None, priors_by_panel=None):
    encoder.eval()
    decoder.eval()
    summary = {}
    with output.open('w', encoding='utf-8') as stream:
        for panel, (covers, records, bank, pairs, clean) in panels.items():
            observed = {name: [] for name in EVALUATION_PATHS}
            identities = []
            all_quality = []
            for offset in range(0, len(pairs), 8):
                ii, jj = zip(*pairs[offset:offset+8])
                targets = [records[i]['identity_id'] for i in ii]
                identities.extend(targets)
                message = bank[list(jj)]
                cover = covers[list(ii)]
                if priors_by_panel is None:
                    image, candidate = watermarked(encoder, cover, message)
                else:
                    image, candidate = watermarked(encoder, cover, message, priors_by_panel[panel][list(ii)])
                current_quality = quality(cover, image, candidate)
                all_quality.extend(current_quality)
                outputs = {'swap': adapter.swap(None, image, opposite(targets), targets),
                           'clean_swap': clean[list(ii)], 'direct': image, 'png': png_roundtrip(image),
                           'jpeg50': transform(image, {'kind': 'jpeg', 'quality': 50}, differentiable=False)}
                labels = message.to(torch.int64).tolist()
                for name, images in outputs.items():
                    logits = decoder(images)
                    require(torch.isfinite(logits).all(), f'Nonfinite logits for {panel}/{name}')
                    prediction = (logits >= 0).to(torch.int64).tolist()
                    swap_rows = [] if observe_swap is not None and panel == 'new' and name == 'swap' else None
                    for j, (label, pred, (cap, clamp, saturated, mse)) in enumerate(zip(labels, prediction, current_quality)):
                        error = sum(x != y for x, y in zip(label, pred, strict=True))
                        observed[name].append(error)
                        record = records[ii[j]]
                        row = dict(panel=panel, path=name, sample_id=record['sample_id'], group_id=record['group_id'],
                                   target_id=targets[j], message_index=jj[j], message=label, predicted_bits=pred,
                                   bit_errors=error, linf=cap, candidate_clipped=clamp, budget_saturated=saturated,
                                   psnr=-10*math.log10(mse) if mse else None)
                        stream.write(json.dumps(row, allow_nan=False) + '\n')
                        if swap_rows is not None:
                            swap_rows.append(row)
                    if swap_rows is not None:
                        observe_swap(images, ii, swap_rows)
            summary[panel] = {}
            for name, errors in observed.items():
                summary[panel][name] = {'all': metrics(errors)}
                summary[panel][name].update({identity: metrics([e for e, i in zip(errors, identities) if i == identity])
                                              for identity in IDENTITIES})
            cap, clamp, saturated, mse = zip(*all_quality, strict=True)
            psnr = [-10*math.log10(value) for value in mse if value]
            summary[panel]['quality'] = dict(max_linf=max(cap), mean_candidate_clipped=sum(clamp)/len(clamp),
                                             mean_budget_saturated=sum(saturated)/len(saturated),
                                             mean_psnr=sum(psnr)/len(psnr) if psnr else None)
    return summary


def run(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    source = read_json(args.checkpoint.parent / 'complete.json')
    hist = read_json(args.historical)
    checkpoint_hash = file_hash(args.checkpoint)
    manifest_hash = file_hash(cfg['assets']['manifest'])
    require(checkpoint_hash == hist['checkpoint_sha256'] == source['checkpoint_sha256'], 'Bad checkpoint hash')
    require(manifest_hash == hist['manifest_sha256'] == design['manifest_sha256'] == source['manifest_sha256'] and
            file_hash(args.design) == hist['design_sha256'], 'Frozen data contract failed')
    require(cfg['budget']['epsilon'] == EPSILON and cfg['model']['message_length'] == 15, 'Bad config contract')
    train, val = (make_dataset(cfg, split=split) for split in ('train', 'val'))
    require(len(train) == 416 and len(val) == 139 and not train.augment and not val.augment, 'Bad split')
    require([r['sample_id'] for r in val.records] == design['val_ids'] and
            all(split_message(m) == 'val' for m in design['val_messages']), 'Bad validation design')
    val_pairs = [tuple(p) for p in design['val_pairs']]
    require(len(val_pairs) == len(set(val_pairs)) == 556 and Counter(i for i, _ in val_pairs) == dict.fromkeys(range(139), 4),
            'Bad validation pairs')
    parent_dir = args.resume.parent.parent if args.resume else None
    parent = read_json(parent_dir / 'complete.json') if parent_dir else None
    parent_plan = read_json(parent_dir / 'plan.json') if parent_dir else None
    start = parent['steps'] if parent else 0
    if parent:
        resume_hash = file_hash(args.resume)
        require((start == 1000 and start < args.steps <= 2000) or (start == 2000 and start < args.steps <= 4000),
                'Resume must extend step 1000 to at most 2000 or step 2000 to at most 4000')
        require(len(parent_plan['batches']) == start and resume_hash == parent['final_checkpoint_sha256'] and
                file_hash(parent_dir / 'plan.json') == parent['plan_sha256'], 'Resume checkpoint or plan hash mismatch')
        require(parent['arm'] == args.arm and parent['seed'] == args.seed and parent['message_recipe'] == args.message_recipe and
                parent['checkpoint_sha256'] == checkpoint_hash and parent['manifest_sha256'] == manifest_hash and
                parent['design_sha256'] == file_hash(args.design), 'Resume recipe or assets differ from the parent')
    if args.coverage_plan:
        from carrier_coverage import COVERAGE_NODES, group_plan
        shared = read_json(args.coverage_plan)
        require((shared['seed'],shared['steps'],shared['manifest_sha256'],shared['design_sha256']) ==
                (args.seed,args.steps,manifest_hash,file_hash(args.design)), 'Coverage plan does not match run inputs')
        selected,plan = group_plan(shared,args.carrier_count,train)
    else:
        selected, plan = prepare_plan(train, design, args.seed, args.steps, parent_plan)
    direct_weight, processed_weight = MESSAGE_WEIGHTS[args.message_recipe]
    if args.message_recipe == 'mixed':
        designated_path = 'geometry_color_clamp' if args.arm == 'j2' else 'swap'
        paths = {'common': 'distortion', 'designated': designated_path}
    else:
        paths = {'common': 'swap', 'designated': 'swap'}
    update_weights = {'direct': direct_weight}
    for path in paths.values():
        update_weights[path] = update_weights.get(path, 0.) + processed_weight/2
    nodes_for_phase = LATER_NODES if start >= 2000 else NODES
    if args.coverage_plan:
        nodes_for_phase = COVERAGE_NODES
    evaluation_steps = {start+node for node in nodes_for_phase if start+node <= args.steps} | {args.steps}
    gradient_middle = start + (1000 if start >= 2000 else 500)
    if args.coverage_plan:
        gradient_middle = 3000
    device = torch.device(args.device)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'plan.json', plan)
    state = torch.load(args.resume or args.checkpoint, weights_only=True, map_location='cpu')
    if parent:
        require(state['step'] == start and state['data_cursor'] == 16*start and
                state['metadata']['plan_sha256'] == parent['plan_sha256'], 'Resume checkpoint cursor mismatch')
    encoder = protocol.OutputEncoder('residual').to(device).eval()
    decoder = MessageDecoder(15).to(device).eval()
    encoder.load_state_dict(state.pop('encoder'), strict=True)
    decoder.load_state_dict(state.pop('decoder'), strict=True)
    adapter = make_adapter(cfg, device).eval().requires_grad_(False)
    validate_clean_quality_report(cfg, adapter)
    require(adapter.checkpoint_hash == hist['adapter_weights_sha256'], 'Wrong Oldpan weights')
    if args.arm == 'j0':
        encoder.requires_grad_(False)
    before = dict(encoder=tensor_digest(encoder), decoder=tensor_digest(decoder), oldpan=tensor_digest(adapter))
    if parent:
        require(before == parent['end_hashes'], 'Resume model differs from historical endpoint')
    initial_bn = [bn_digest(module) for module in (encoder, decoder)]
    train_records = [train.records[i] for i in selected]
    train_ids = [r['identity_id'] for r in train_records]
    covers = torch.stack([train[i]['target'] for i in selected]).to(device)
    val_covers = torch.stack([val[i]['target'] for i in range(len(val))]).to(device)
    metadata = dict(arm=args.arm, seed=args.seed, steps=args.steps, device=str(device), gpu=torch.cuda.get_device_name(device),
                    script_sha256=file_hash(__file__), checkpoint_sha256=checkpoint_hash, plan_sha256=file_hash(args.output / 'plan.json'),
                    manifest_sha256=manifest_hash, design_sha256=file_hash(args.design), oldpan_sha256=adapter.checkpoint_hash,
                    start_hashes=before, batch_size=16, micro_batch=8, lr=5e-6, processed_weight=processed_weight,
                    message_recipe=args.message_recipe, processed_paths=paths, message_weights_per_update=update_weights,
                    encoder_bn_stats_frozen=True, decoder_bn_stats_frozen=True, epsilon=EPSILON,
                    training_carriers=len(selected), same_pairs=512, new_pairs=556, test_images_read=0, torch_version=str(torch.__version__))
    if args.coverage_plan:
        metadata.update(coverage_plan_sha256=file_hash(args.coverage_plan), carrier_count=args.carrier_count,
                        message_frequency_histogram=shared['message_frequency_histogram'],
                        training_pair_observations=16*args.steps)
    if parent:
        old_messages = {tuple(m) for batch in plan['batches'][:start] for micro in batch['micros'] for m in micro['message']}
        tail_messages = {tuple(m) for batch in plan['batches'][start:] for micro in batch['micros'] for m in micro['message']}
        metadata.update(resume_from_step=start, parent_checkpoint_sha256=resume_hash, parent_plan_sha256=parent['plan_sha256'],
                        continuation_new_messages=len(tail_messages-old_messages),
                        continuation_recycled_messages=len(tail_messages & old_messages))
        if start >= 2000:
            counts = Counter(tuple(m) for batch in plan['batches'] for micro in batch['micros'] for m in micro['message'])
            metadata.update(continuation_observations=16*(args.steps-start), message_frequency_histogram=dict(Counter(counts.values())))
    write_json(args.output / 'config.json', metadata)
    probes = {}
    if args.arm == 'j1':
        probes[str(start)] = gradient_probe(encoder, decoder, adapter, covers, train_ids, plan, device)
        write_json(args.output / 'gradients.json', probes)
        print('J1 swap BCE preflight', json.dumps(probes[str(start)]), flush=True)
    panels = {
        'same': (covers[:16], train_records[:16], torch.tensor(plan['same_messages'], device=device, dtype=torch.float32),
                 [(i,j) for i in range(16) for j in range(32)], clean_outputs(adapter, covers[:16], train_ids[:16])),
        'new': (val_covers, val.records, torch.tensor(design['val_messages'], device=device, dtype=torch.float32),
                val_pairs, clean_outputs(adapter, val_covers, [r['identity_id'] for r in val.records])),
    }
    parameters = list(decoder.parameters()) + [p for p in encoder.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(parameters, lr=5e-6, weight_decay=1e-5)
    if parent:
        optimizer.load_state_dict(state.pop('optimizer'))
        require(bool(optimizer.state) and all(int(slot['step']) == start for slot in optimizer.state.values()),
                'Adam step counters were not restored')
        require(all(group['lr'] == 5e-6 and group['weight_decay'] == 1e-5 for group in optimizer.param_groups),
                'Resume optimizer hyperparameters differ')
    nodes = []
    (args.output / 'checkpoints').mkdir()
    if not parent:
        torch.manual_seed(args.seed)

    def save_node(step):
        result = evaluate(encoder, decoder, adapter, panels, args.output / f'validation-{step:04d}.jsonl')
        if parent and step == start:
            historical = read_json(parent_dir / 'evaluation.json')[-1]
            require(historical['step'] == start and historical['panels'] == result and
                    file_hash(args.output / f'validation-{step:04d}.jsonl') == file_hash(parent_dir / f'validation-{step:04d}.jsonl'),
                    'Resume baseline prediction replay mismatch')
            torch.set_rng_state(state.pop('cpu_rng'))
            torch.cuda.set_rng_state(state.pop('cuda_rng'), device=device)
            random.setstate(state.pop('python_rng'))
            print('Restored original validation predictions, Adam state, RNG and cursor', flush=True)
        nodes.append(dict(step=step, panels=result))
        write_json(args.output / 'evaluation.json', nodes)
        torch.save(dict(step=step, data_cursor=16*step, encoder=encoder.state_dict(), decoder=decoder.state_dict(),
                        optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state(device),
                        python_rng=random.getstate(), metadata=metadata), args.output / 'checkpoints' / f'step-{step:04d}.pt')
        print(json.dumps(dict(step=step, same=result['same']['swap'], new=result['new']['swap']['all'],
                              quality=result['same']['quality'])), flush=True)
        return result

    base = save_node(start)
    if not parent:
        for path in ('direct', 'swap'):
            for scope in ('all', *IDENTITIES):
                a, b = base['new'][path][scope], hist['by_path'][path][scope]
                require(a['exact'] == b['exact'] and abs(a['ber']-b['ber']) < 1e-12, 'Baseline replay mismatch')
    with (args.output / 'train.jsonl').open('w', encoding='utf-8', buffering=1) as stream:
        for step, batch in enumerate(plan['batches'][start:], start+1):
            encoder.train(args.arm != 'j0')
            decoder.train()
            freeze_bn(encoder)
            freeze_bn(decoder)
            optimizer.zero_grad(set_to_none=True)
            logs = dict(step=step)
            for micro in batch['micros']:
                indices, labels = micro['indices'], micro['message']
                cover, message = covers[indices], torch.tensor(labels, device=device, dtype=torch.float32)
                identities = [train_ids[i] for i in indices]
                image, candidate = watermarked(encoder, cover, message)
                stats = quality(cover, image, candidate)
                path = paths[micro['slot']]
                if path == 'distortion':
                    processed = transform(image, batch['transform'], differentiable=True)
                elif path == 'geometry_color_clamp':
                    processed = adapter.restore_image(adapter.native_input(image, identities)).clamp(0, 1)
                else:
                    processed = adapter.swap(None, image, opposite(identities), identities)
                direct_logits, processed_logits = decoder(torch.cat((image, processed))).chunk(2)
                direct_bce = F.binary_cross_entropy_with_logits(direct_logits, message)
                processed_bce = F.binary_cross_entropy_with_logits(processed_logits, message)
                mse = F.mse_loss(image, cover)
                structure = image_structure_loss(image, cover)
                loss = mse + .1*structure + direct_weight*direct_bce + processed_weight*processed_bce
                require(torch.isfinite(loss), 'Nonfinite training loss')
                (loss/2).backward()
                logs[micro['slot']] = dict(processed_path=path, loss=float(loss.detach()), direct_bce=float(direct_bce.detach()),
                                           processed_bce=float(processed_bce.detach()), mse=float(mse.detach()),
                                           structure=float(structure.detach()), max_linf=max(s[0] for s in stats),
                                           candidate_clipped=sum(s[1] for s in stats)/8,
                                           budget_saturated=sum(s[2] for s in stats)/8)
            grad = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
            logs['combined_grad_norm'] = float(grad)
            optimizer.step()
            stream.write(json.dumps(logs, allow_nan=False) + '\n')
            if step in evaluation_steps:
                save_node(step)
                if args.arm == 'j1' and step in (gradient_middle, args.steps):
                    probes[str(step)] = gradient_probe(encoder, decoder, adapter, covers, train_ids, plan, device)
                    write_json(args.output / 'gradients.json', probes)
            elif step % 50 == 0:
                print(f'completed {step} updates, designated processed BCE {logs["designated"]["processed_bce"]:.5f}', flush=True)
    after = dict(encoder=tensor_digest(encoder), decoder=tensor_digest(decoder), oldpan=tensor_digest(adapter))
    require(after['oldpan'] == before['oldpan'] and all(p.grad is None for p in adapter.parameters()), 'Oldpan changed')
    require((after['encoder'] == before['encoder']) == (args.arm == 'j0') and after['decoder'] != before['decoder'],
            'Updated modules do not match arm')
    require(all(torch.equal(a, b) and n == m for module, frozen in zip((encoder, decoder), initial_bn, strict=True)
                for (n, a), (m, b) in zip(bn_digest(module), frozen, strict=True)), 'BN buffers changed')
    write_json(args.output / 'complete.json', dict(metadata, end_hashes=after, gradient_probes=probes,
               final_checkpoint_sha256=file_hash(args.output / 'checkpoints' / f'step-{args.steps:04d}.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assets', 'design', 'checkpoint', 'historical', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--arm', required=True, choices=('j0','j1','j2'))
    parser.add_argument('--message-recipe', choices=MESSAGE_WEIGHTS, default='mixed',
                        help='Use mixed processing, all FaceSwap at 75/25, or all FaceSwap at 50/50')
    parser.add_argument('--steps', type=int, default=1000, help='Absolute final optimizer step')
    parser.add_argument('--resume', type=Path, help='Verified step-1000.pt or step-2000.pt from a complete run')
    parser.add_argument('--coverage-plan', type=Path, help='Frozen common plan for a carrier-coverage comparison')
    parser.add_argument('--carrier-count', type=int, choices=(16,64,416))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()
    if args.resume and not 1001 <= args.steps <= 4000:
        parser.error('--resume requires an absolute final step between 1001 and 4000')
    if bool(args.coverage_plan) != bool(args.carrier_count):
        parser.error('--coverage-plan and --carrier-count must be supplied together')
    if args.coverage_plan and (args.resume or args.arm != 'j1' or args.message_recipe == 'mixed' or not 1 <= args.steps <= 6000):
        parser.error('Coverage runs require a fresh J1 full-swap run of at most 6000 updates')
    if not args.resume and not args.coverage_plan and not 1 <= args.steps <= 1000:
        parser.error('without --resume, --steps must be between 1 and 1000')
    if args.arm == 'j2' and args.message_recipe != 'mixed':
        parser.error('j2 is the network-bypass control and requires the mixed recipe')
    run(args)
