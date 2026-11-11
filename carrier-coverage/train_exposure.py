"""Resume a verified carrier-coverage endpoint under a frozen exposure schedule."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from exposure_plan import MILESTONES, START
from run_joint import (EPSILON, IDENTITIES, MESSAGE_WEIGHTS, MessageDecoder, bn_digest, clean_outputs, evaluate,
                       freeze_bn, image_structure_loss, make_adapter, make_dataset, protocol,
                       quality, tensor_digest, validate_clean_quality_report, watermarked)
from rgwm.config import load_run_config
from rgwm.data.messages import split_message
from rgwm.legacy import file_hash
from run_probe import opposite, read_json, require, write_json


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    write_json(temporary, data)
    temporary.replace(path)


def paired_losses(cover, image, labels, direct_logits, swap_logits, indices):
    return (F.mse_loss(image[indices], cover[indices]),
            image_structure_loss(image[indices], cover[indices]),
            F.binary_cross_entropy_with_logits(direct_logits[indices], labels[indices]),
            F.binary_cross_entropy_with_logits(swap_logits[indices], labels[indices]))


def validate_ofea_weight(weight, recipe):
    require(math.isfinite(weight) and weight >= 0, 'OFEA weight must be finite and nonnegative')
    require(weight == 0 or recipe == 'swap-equal', 'OFEA is only enabled for the 50:50 recipe')


def fork_metadata(args, recipe, metadata):
    validate_ofea_weight(args.ofea_weight, recipe)
    if args.resume_from is None:
        require(args.ofea_weight == 0, 'Use --resume-from to run an OFEA comparison')
        require(args.through_step in MILESTONES[args.carriers], 'Requested stop is not a frozen milestone')
        return None, None, metadata

    require(recipe == 'swap-equal', 'Exposure forks require the 50:50 recipe')
    source = args.resume_from.resolve(strict=True)
    require(args.output.resolve() != source.parent, 'Use a separate output directory for the fork')
    step_text = source.stem.removeprefix('step-')
    require(source.name.startswith('step-') and source.suffix == '.pt' and step_text.isdecimal(),
            'Fork source must be a saved validation checkpoint (step-<step>.pt)')
    step = int(step_text)
    require(step in MILESTONES[args.carriers] and step < args.through_step <= MILESTONES[args.carriers][-1],
            'Fork must start at a validated checkpoint and use the remaining frozen schedule')
    source_config = read_json(source.parent / 'config.json')
    require(source_config == dict(metadata, device=source_config.get('device')),
            'Fork checkpoint comes from a different exposure plan or recipe')
    require((source.parent / f'validation-{step}.jsonl').is_file() and
            (source.parent / f'evaluation-{step}.json').is_file(), 'Fork source needs matching validation records')
    nodes = sorted({step, args.through_step} | {step+delta for delta in (250,500) if step+delta < args.through_step})
    fork = dict(metadata, nodes=nodes, fork_step=step, fork_checkpoint=str(source),
                fork_checkpoint_sha256=file_hash(source), ofea_weight=args.ofea_weight)
    return source, source_config, fork


def update(encoder, decoder, adapter, optimizer, parameters, covers, ids, indices, bits, step, message_weights,
           *, ofea_weight=0., track_ofea=False, priors=None):
    totals = Counter(ids[i] for i in indices)
    require(set(totals) == set(IDENTITIES), 'Identity missing from update')
    optimizer.zero_grad(set_to_none=True)
    logs = dict(step=step, identity_counts=dict(totals))
    for start, slot in ((0, 'designated'), (8, 'common')):
        carrier_ids = indices[start:start+8]
        cover, message = covers[carrier_ids], bits[start:start+8]
        targets = [ids[i] for i in carrier_ids]
        if priors is None:
            protected, candidate = watermarked(encoder, cover, message)
        else:
            protected, candidate = watermarked(encoder, cover, message, priors[carrier_ids])
        stats = quality(cover, protected, candidate)
        if ofea_weight or track_ofea:
            pair = adapter.forward_pair(protected, source_ids=opposite(targets), target_ids=targets)
            swapped = pair.image
        else:
            swapped = adapter.swap(None, protected, opposite(targets), targets)
        direct_logits, swap_logits = decoder(torch.cat((protected, swapped))).chunk(2)
        by_id = {name:[i for i,t in enumerate(targets) if t == name] for name in IDENTITIES}
        weights = {name:len(by_id[name])/(2*totals[name]) for name in IDENTITIES}
        if totals['cage'] == 8:
            losses = paired_losses(cover, protected, message, direct_logits, swap_logits, slice(None))
            terms = [value/2 for value in losses]
        else:
            weighted = [[weights[name]*value for value in paired_losses(
                cover, protected, message, direct_logits, swap_logits, by_id[name])]
                        for name in IDENTITIES if by_id[name]]
            terms = [sum(values) for values in zip(*weighted, strict=True)]
        mse, structure, direct, swap = terms
        loss = mse + .1*structure + message_weights['direct']*direct + message_weights['swap']*swap
        if ofea_weight or track_ofea:
            # Native Oldpan pre-sigmoid features from the same watermarked carrier.
            with torch.set_grad_enabled(bool(ofea_weight)):
                gap = (pair.target_logits.float() - pair.source_logits.float()).square().flatten(1).mean(1)
                if totals['cage'] == 8:
                    ofea = gap.mean() / 2
                else:
                    ofea = sum(weights[name] * gap[by_id[name]].mean() for name in IDENTITIES if by_id[name])
                weighted_ofea = ofea_weight * ofea
            require(torch.isfinite(ofea), 'Nonfinite OFEA loss')
            if ofea_weight:
                loss = loss + weighted_ofea
        require(torch.isfinite(loss), 'Nonfinite training loss')
        loss.backward()
        logs[slot] = dict(processed_path='swap', identity_weights=weights, contribution=float(loss.detach()),
                          mse=float(mse.detach()), structure=float(structure.detach()),
                          direct_bce=float(direct.detach()), swap_bce=float(swap.detach()),
                          max_linf=max(v[0] for v in stats))
        if ofea_weight or track_ofea:
            logs[slot].update(ofea=float(ofea.detach()), weighted_ofea=float(weighted_ofea.detach()))
    gradient = torch.nn.utils.clip_grad_norm_(parameters, float('inf'), error_if_nonfinite=True)
    logs['combined_grad_norm'] = float(gradient)
    optimizer.step()
    return logs


def probe_ofea(encoder, decoder, adapter, covers, ids, batch, bits, *, priors=None, freeze_running=True):
    """Measure the encoder gradient scale at the first scheduled fork update."""
    for model in (encoder,decoder):
        model.train()
        if freeze_running:
            freeze_bn(model)
    parameters = tuple(encoder.parameters())
    result = {}
    for identity in IDENTITIES:
        chosen = [i for i,carrier in enumerate(batch) if ids[carrier] == identity][:4]
        require(bool(chosen), 'OFEA probe requires both identities')
        carriers = [batch[i] for i in chosen]
        targets = [identity]*len(chosen)
        labels = bits[chosen]
        if priors is None:
            protected, _ = watermarked(encoder,covers[carriers],labels)
        else:
            protected, _ = watermarked(encoder,covers[carriers],labels,priors[carriers])
        pair = adapter.forward_pair(protected,source_ids=opposite(targets),target_ids=targets)
        direct, swap = decoder(torch.cat((protected,pair.image))).chunk(2)
        message = (F.binary_cross_entropy_with_logits(direct,labels) +
                   F.binary_cross_entropy_with_logits(swap,labels)) / 2
        ofea = F.mse_loss(pair.target_logits.float(),pair.source_logits.float())
        norms = {}
        for name,loss in (('message',message),('ofea',ofea)):
            gradients = torch.autograd.grad(loss,parameters,retain_graph=name == 'message',allow_unused=True)
            norm = math.sqrt(sum(float(g.detach().float().square().sum()) for g in gradients if g is not None))
            require(math.isfinite(norm) and norm > 0, f'{identity} {name} gradient is missing or nonfinite')
            norms[name] = norm
        result[identity] = norms
    recommended = .1 * min(v['message']/v['ofea'] for v in result.values())
    return dict(encoder_gradient_norms=result, suggested_ofea_weight=recommended,
                target_max_weighted_gradient_ratio=.1)


def run(args, resources):
    size = args.carriers
    plan_path = args.plan_root / f'n{size}-schedule.npz'
    schedule = read_json(args.plan_root / f'n{size}-schedule.json')
    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    parent = args.parent / f'n{size}-seed0'
    original = read_json(parent / 'complete.json')
    recipe = original['message_recipe']
    require(recipe in ('swap-weighted','swap-equal'), 'Expected a full-swap parent')
    message_weights = dict(zip(('direct','swap'), MESSAGE_WEIGHTS[recipe], strict=True))
    start_checkpoint = parent / 'checkpoints' / f'step-{START}.pt'
    require(file_hash(start_checkpoint) == original['final_checkpoint_sha256'], 'Wrong parent checkpoint')
    require(file_hash(plan_path) == schedule['plan_sha256'] and schedule['size'] == size and schedule['start'] == START,
            'Invalid continuation schedule')
    require(schedule['parent_plan_sha256'] == original['plan_sha256'] == file_hash(parent / 'plan.json') and
            schedule['manifest_sha256'] == original['manifest_sha256'] == file_hash(cfg['assets']['manifest']) and
            schedule['design_sha256'] == original['design_sha256'] == file_hash(args.design) and
            cfg['budget']['epsilon'] == EPSILON and cfg['model']['message_length'] == 15 and
            original['steps'] == START and original['message_weights_per_update'] == message_weights,
            'Frozen run inputs differ')
    with np.load(plan_path, allow_pickle=False) as npz:
        indices, codes = npz['carriers'].astype(np.int64), npz['messages']
    require(indices.shape == codes.shape == (MILESTONES[size][-1]-START,16), 'Wrong schedule dimensions')
    train, val = (make_dataset(cfg, split=split) for split in ('train','val'))
    require(len(train) == 416 and len(val) == 139 and not train.augment and not val.augment, 'Bad split')
    require([r['sample_id'] for r in val.records] == design['val_ids'] and
            all(split_message(m) == 'val' for m in design['val_messages']), 'Validation assets changed')
    old_plan = read_json(parent / 'plan.json')
    require(old_plan['train_ids'] == schedule['train_ids'], 'Carrier order changed')
    train_lookup = {r['sample_id']:i for i,r in enumerate(train.records)}
    positions = [train_lookup[name] for name in schedule['train_ids']]
    train_records = [train.records[i] for i in positions]
    ids = [r['identity_id'] for r in train_records]
    val_pairs = [tuple(p) for p in design['val_pairs']]
    require(len(val_pairs) == len(set(val_pairs)) == 556, 'Bad validation pairs')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    encoder = protocol.OutputEncoder('residual').to(device).eval()
    decoder = MessageDecoder(15).to(device).eval()
    adapter = make_adapter(cfg, device).eval().requires_grad_(False)
    clean_report = validate_clean_quality_report(cfg, adapter)
    oldpan = tensor_digest(adapter)
    require(adapter.checkpoint_hash == original['oldpan_sha256'] and oldpan == original['end_hashes']['oldpan'],
            'Oldpan differs from the parent')
    covers = torch.stack([train[i]['target'] for i in positions]).to(device)
    val_covers = torch.stack([val[i]['target'] for i in range(len(val))]).to(device)
    panels = {
        'same': (covers[:16], train_records[:16], torch.tensor(old_plan['same_messages'],device=device,dtype=torch.float32),
                 [(i,j) for i in range(16) for j in range(32)], clean_outputs(adapter,covers[:16],ids[:16])),
        'new': (val_covers, val.records, torch.tensor(design['val_messages'],device=device,dtype=torch.float32),
                val_pairs, clean_outputs(adapter,val_covers,[r['identity_id'] for r in val.records])),
    }
    parameters = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(parameters, lr=5e-6, weight_decay=1e-5)
    output = args.output
    latest = output / 'latest.pt'
    metadata = dict(plan_sha256=schedule['plan_sha256'], parent_sha256=original['final_checkpoint_sha256'],
                    start=START, carriers=size, manifest_sha256=schedule['manifest_sha256'],
                    design_sha256=schedule['design_sha256'], oldpan_sha256=original['oldpan_sha256'],
                    message_weights=message_weights, identity_weights={'cage':.5,'trump':.5},
                    nodes=[START,*MILESTONES[size]], test_images_read=0, device=str(device))
    fork, fork_config, metadata = fork_metadata(args, recipe, metadata)
    if args.probe_ofea:
        require(fork and not args.identity_pairs, '--probe-ofea requires a fork and runs without identity evaluation')
    identity_observer = None
    if args.identity_pairs:
        from exposure_identity import prepare_identity_observer
        identity_observer = prepare_identity_observer(cfg,adapter,clean_report,args.identity_pairs,panels['new'],resources)
        metadata['identity_pairs_sha256'] = file_hash(args.identity_pairs)
    first_step = metadata.get('fork_step', START)
    validation_steps = set(metadata['nodes'])
    if not args.probe_ofea:
        output.mkdir(parents=True, exist_ok=True)
        path = output / 'config.json'
        if path.exists():
            require(read_json(path) == metadata, 'Existing continuation differs from this job')
        else:
            atomic_json(path, metadata)
    resuming = latest.exists() and not args.probe_ofea
    state = torch.load(latest if resuming else fork or start_checkpoint, map_location='cpu', weights_only=True)
    start = state['step']
    log_offset = state['log_offset'] if resuming else 0
    require(state['data_cursor'] == start*16 and first_step <= start <= args.through_step and
            (state['metadata'] == metadata if resuming else start == first_step), 'Checkpoint cursor mismatch')
    if fork and not resuming:
        require(state['metadata'] == fork_config, 'Fork checkpoint metadata differs from its source configuration')
    if resuming and start in validation_steps and not (output / f'step-{start}.pt').exists():
        os.link(latest, output / f'step-{start}.pt')
    encoder.load_state_dict(state['encoder'], strict=True)
    decoder.load_state_dict(state['decoder'], strict=True)
    # Buffers from the parent checkpoint are compared throughout the continuation.
    initial_bn = [bn_digest(model) for model in (encoder, decoder)]
    if start == START:
        for name, model in (('encoder',encoder),('decoder',decoder)):
            require(tensor_digest(model) == original['end_hashes'][name], f'Wrong parent {name}')
    optimizer.load_state_dict(state['optimizer'])
    require(len(optimizer.state) == len(parameters) and all(int(s['step']) == start for s in optimizer.state.values()) and
            all(g['lr'] == 5e-6 and g['weight_decay'] == 1e-5 for g in optimizer.param_groups), 'Adam resume mismatch')
    labels = ((torch.arange(1 << 15,device=device)[:,None] >> torch.arange(14,-1,-1,device=device)) & 1).float()
    if args.probe_ofea:
        index = start-START
        result = probe_ofea(encoder,decoder,adapter,covers,ids,indices[index].tolist(),labels[codes[index].astype(np.int64)])
        print(json.dumps(dict(step=start+1, **result),allow_nan=False),flush=True)
        return

    def evaluate_node(step):
        temp = output / f'validation-{step}.tmp'
        identity_rows = []

        def record_identity(images, indices, rows):
            identity_rows.extend(identity_observer(images, indices, rows))

        stats = evaluate(encoder, decoder, adapter, panels, temp,
                         observe_swap=record_identity if identity_observer else None)
        if step == first_step:
            source_folder = fork.parent if fork else parent
            source_stats = (read_json(source_folder / f'evaluation-{step}.json') if fork
                            else read_json(source_folder / 'evaluation.json')[-1])
            require(file_hash(temp) == file_hash(source_folder / f'validation-{step}.jsonl') and
                    stats == source_stats['panels'], 'Parent predictions differ')
        temp.replace(output / f'validation-{step}.jsonl')
        evaluation = dict(step=step, panels=stats)
        if identity_observer:
            from exposure_identity import identity_summary
            require(len(identity_rows) == len(val_pairs), 'Identity observations differ from validation pairs')
            identity_temp = output / f'identity-{step}.tmp'
            with identity_temp.open('w', encoding='utf-8') as stream:
                for row in identity_rows:
                    stream.write(json.dumps(row, allow_nan=False) + '\n')
            identity_temp.replace(output / f'identity-{step}.jsonl')
            evaluation['identity'] = identity_summary(identity_rows)
        atomic_json(output / f'evaluation-{step}.json', evaluation)
        print(json.dumps(dict(step=step,new=stats['new']['swap']['all'],same=stats['same']['swap']['all'])), flush=True)

    if start == first_step:
        evaluate_node(start)
    torch.set_rng_state(state['cpu_rng'])
    torch.cuda.set_rng_state(state['cuda_rng'], device=device)
    random.setstate(state['python_rng'])
    del state
    print(f'N{size} resumed step {start} with Adam and cursor {start*16}', flush=True)
    begin = time.monotonic()
    logfile = output / 'train.jsonl'
    with logfile.open('a+b') as stream:
        require(stream.tell() >= log_offset, 'Training log truncated before durable checkpoint')
        stream.truncate(log_offset)
        stream.seek(log_offset)
        for step in range(start+1,args.through_step+1):
            for model in (encoder,decoder):
                model.train()
                freeze_bn(model)
            index = step-START-1
            batch = indices[index].tolist()
            message = labels[codes[index].astype(np.int64)]
            row = update(encoder,decoder,adapter,optimizer,parameters,covers,ids,batch,message,step,message_weights,
                         ofea_weight=args.ofea_weight,track_ofea=bool(fork))
            stream.write((json.dumps(row,allow_nan=False)+'\n').encode('utf-8'))
            if step in validation_steps:
                evaluate_node(step)
            if step % 1000 == 0 or step in validation_steps:
                stream.flush()
                os.fsync(stream.fileno())
                require(all(int(s['step']) == step for s in optimizer.state.values()), 'Adam step counter mismatch')
                require(all(n == m and torch.equal(a,b)
                            for model, frozen in zip((encoder,decoder),initial_bn,strict=True)
                            for (n,a),(m,b) in zip(bn_digest(model),frozen,strict=True)), 'BN buffer changed')
                temp = output / 'latest.tmp'
                torch.save(dict(step=step, data_cursor=16*step, log_offset=stream.tell(), encoder=encoder.state_dict(),
                                decoder=decoder.state_dict(), optimizer=optimizer.state_dict(), cpu_rng=torch.get_rng_state(),
                                cuda_rng=torch.cuda.get_rng_state(device), python_rng=random.getstate(), metadata=metadata), temp)
                temp.replace(latest)
                if step in validation_steps:
                    stage = output / f'step-{step}.pt'
                    require(not stage.exists(), 'Milestone checkpoint already exists')
                    os.link(latest,stage)
            if step % 100 == 0:
                stream.flush()
                rate = (step-start)/(time.monotonic()-begin)
                print(f'N{size} {step}/{args.through_step} at {rate:.2f} updates/s', flush=True)
    require(tensor_digest(adapter) == oldpan and all(p.grad is None for p in adapter.parameters()), 'Oldpan changed')
    endpoint = output / f'step-{args.through_step}.pt'
    atomic_json(output / f'final-{args.through_step}.json', dict(metadata, step=args.through_step,
                data_cursor=args.through_step*16, final_checkpoint_sha256=file_hash(endpoint),
                end_hashes=dict(encoder=tensor_digest(encoder),decoder=tensor_digest(decoder),oldpan=oldpan)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','design','parent','plan-root','output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--carriers', type=int, choices=tuple(MILESTONES), required=True)
    parser.add_argument('--through-step', type=int, required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--resume-from', type=Path, help='Verified 50:50 exposure checkpoint to fork')
    parser.add_argument('--ofea-weight', type=float, default=0., help='Weight of native pre-sigmoid feature MSE')
    parser.add_argument('--identity-pairs', type=Path, help='Accepted validation pairs for joint identity evaluation')
    parser.add_argument('--probe-ofea', action='store_true', help='Measure encoder gradients and exit before training')
    with ExitStack() as resources:
        run(parser.parse_args(), resources)
