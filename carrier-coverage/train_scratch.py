"""Train the two-head N416 watermark model from a shared random initialization."""

import argparse
from collections import Counter
from contextlib import ExitStack
import json
import os
from pathlib import Path
import random
from statistics import median
import time

import numpy as np
import torch

from exposure_identity import identity_summary, prepare_identity_observer
from exposure_plan import START, message_id
from rgwm.models.factory import build_encoder
from rgwm.training.trainer import preserve_buffers
from run_joint import (EPSILON, MessageDecoder, clean_outputs, evaluate, make_adapter,
                       make_dataset, tensor_digest, validate_clean_quality_report)
from train_exposure import (atomic_json, file_hash, load_run_config, probe_ofea,
                            read_json, require, split_message, update)


STEPS = 156000
NODES = (0, 1000, 6000, 13000, 26000, 39000, 78000, 117000, STEPS)


def load_schedule(root, manifest_hash, design_hash):
    parent = root / 'n416-seed0' / 'plan.json'
    folder = root / 'exposure'
    plan = folder / 'n416-schedule.npz'
    meta = read_json(folder / 'n416-schedule.json')
    require(meta['manifest_sha256'] == manifest_hash and meta['design_sha256'] == design_hash and
            meta['parent_plan_sha256'] == file_hash(parent) and meta['plan_sha256'] == file_hash(plan),
            'Frozen N416 schedule differs from the current data')
    initial = read_json(parent)
    require(initial['train_ids'] == meta['train_ids'] and len(initial['batches']) == START, 'Bad initial N416 schedule')
    for batch in initial['batches']:
        require([micro['slot'] for micro in batch['micros']] == ['designated','common'], 'Changed microbatch order')
    indices = [[i for micro in b['micros'] for i in micro['indices']] for b in initial['batches']]
    codes = [[message_id(m) for micro in b['micros'] for m in micro['message']] for b in initial['batches']]
    with np.load(plan, allow_pickle=False) as saved:
        indices = np.concatenate((np.asarray(indices),saved['carriers']),axis=0)
        codes = np.concatenate((np.asarray(codes),saved['messages']),axis=0)
    require(indices.shape == codes.shape == (STEPS,16) and len(set(meta['train_ids'])) == 416, 'Wrong N416 dimensions')
    require(np.all((0 <= indices) & (indices < 416)) and
            np.all(np.bincount(indices.ravel(),minlength=416) == 6000), 'Bad carrier exposures')
    require(np.unique((indices << 15)+codes).size == codes.size, 'Repeated carrier-message pairs')
    require(all(split_message([int(bit) for bit in f'{code:015b}']) == 'train'
                for code in np.unique(codes)), 'Nontraining messages in the schedule')
    return meta, initial['same_messages'], indices, codes


def load_images(dataset, order, device):
    samples = [dataset[i] for i in order]
    return (torch.stack([s['target'] for s in samples]).to(device),
            torch.stack([s['priors'] for s in samples]).to(device))


def calibration(encoder, decoder, adapter, covers, priors, ids, indices, codes, labels, device):
    with torch.random.fork_rng(devices=[device.index]), preserve_buffers(encoder), preserve_buffers(decoder):
        probes = [probe_ofea(encoder,decoder,adapter,covers,ids,indices[i].tolist(),labels[codes[i]],
                             priors=priors,freeze_running=False) for i in range(4)]
    directions = ('cage','trump')
    ratios = {name:median(p['encoder_gradient_norms'][name]['message'] /
                         p['encoder_gradient_norms'][name]['ofea'] for p in probes) for name in directions}
    weight = .1 * min(ratios.values())
    require(np.isfinite(weight) and weight > 0, 'Bad OFEA calibration')
    return dict(probes=probes, median_message_to_ofea=ratios, ofea_weight=weight)


def run(args, resources):
    cfg = load_run_config(args.config,args.assets.resolve())
    design = read_json(args.design)
    manifest_hash = file_hash(cfg['assets']['manifest'])
    design_hash = file_hash(args.design)
    require(cfg['model']['variant'] == 'M' and cfg['model']['message_length'] == 15 and
            cfg['budget']['epsilon'] == EPSILON and design['manifest_sha256'] == manifest_hash, 'Wrong M configuration')
    plan, same_messages, indices, codes = load_schedule(args.baseline,manifest_hash,design_hash)
    train, val = (make_dataset(cfg,split=split) for split in ('train','val'))
    require(len(train) == 416 and len(val) == 139 and not train.augment and not val.augment, 'Bad carrier split')
    require([r['sample_id'] for r in val.records] == design['val_ids'] and
            all(split_message(m) == 'val' for m in (*same_messages,*design['val_messages'])), 'Bad held-out design')
    by_id = {r['sample_id']:i for i,r in enumerate(train.records)}
    order = [by_id[i] for i in plan['train_ids']]
    ids = [train.records[i]['identity_id'] for i in order]
    require(Counter(ids) == {'cage':190,'trump':226}, 'Bad N416 identity counts')
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    adapter = make_adapter(cfg,device).eval().requires_grad_(False)
    clean_report = validate_clean_quality_report(cfg,adapter)
    pairs_hash = file_hash(args.identity_pairs)
    require(clean_report and clean_report['pair_manifest_hash'] == pairs_hash, 'Bad identity reference manifest')
    oldpan = tensor_digest(adapter)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    encoder = build_encoder('M',message_length=15,epsilon=EPSILON,
                            head_channels=tuple(cfg['model']['head_channels'])).to(device)
    decoder = MessageDecoder(15).to(device)
    initial = {name:tensor_digest(module) for name,module in (('encoder',encoder),('decoder',decoder))}
    parameters = list(decoder.parameters()) + list(encoder.parameters())
    optimizer = torch.optim.Adam(parameters,lr=5e-5,weight_decay=1e-5)
    require(not optimizer.state, 'Adam must start empty')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=STEPS)
    covers, priors = load_images(train,order,device)
    val_covers, val_priors = load_images(val,range(len(val)),device)
    labels = ((torch.arange(1<<15,device=device)[:,None] >> torch.arange(14,-1,-1,device=device)) & 1).float()
    val_pairs = [tuple(p) for p in design['val_pairs']]
    require(len(set(val_pairs)) == 556 and Counter(i for i,_ in val_pairs) == dict.fromkeys(range(139),4),
            'Bad fixed validation pairs')
    panels = {
        'same':(covers[:16],[train.records[i] for i in order[:16]],
                torch.tensor(same_messages,device=device,dtype=torch.float32),
                [(i,j) for i in range(16) for j in range(32)],clean_outputs(adapter,covers[:16],ids[:16])),
        'new':(val_covers,val.records,torch.tensor(design['val_messages'],device=device,dtype=torch.float32),
               val_pairs,clean_outputs(adapter,val_covers,[r['identity_id'] for r in val.records])),
    }
    panel_priors = {'same':priors[:16],'new':val_priors}
    root = args.output
    root.mkdir(parents=True,exist_ok=True)
    latest = root / 'latest.pt'
    probe_path = root / 'calibration.json'
    weight = 0.
    if args.arm == 'ofea':
        if not probe_path.exists():
            require(not latest.exists(), 'Missing calibration for saved OFEA run')
            atomic_json(probe_path,calibration(encoder,decoder,adapter,covers,priors,ids,indices,codes,labels,device))
        weight = read_json(probe_path)['ofea_weight']
        require(np.isfinite(weight) and weight > 0, 'Invalid stored OFEA calibration')
    meta = dict(architecture='M',init='random',initial_hashes=initial,arm=args.arm,seed=args.seed,
                device=str(device),manifest_sha256=manifest_hash,design_sha256=design_hash,
                first_plan_sha256=plan['parent_plan_sha256'],continuation_plan_sha256=plan['plan_sha256'],
                ofea_weight=weight,ofea_ramp_steps=2000,lr=5e-5,weight_decay=1e-5,steps=STEPS,
                nodes=list(NODES),batch_size=16,precision='fp32',message_weights={'direct':.5,'swap':.5},epsilon=EPSILON,
                oldpan=oldpan,identity_pairs_sha256=pairs_hash,test_images_read=0)
    if (root / 'config.json').exists():
        require(read_json(root / 'config.json') == meta, 'Scratch configuration changed')
    else:
        atomic_json(root / 'config.json',meta)
    print(json.dumps(dict(initial_hashes=initial,ofea_weight=weight,device=str(device))),flush=True)

    start, offset = 0, 0
    if latest.exists():
        state = torch.load(latest,weights_only=True,map_location='cpu')
        start, offset = state['step'], state['log_offset']
        require(state['metadata'] == meta and state['data_cursor'] == 16*start, 'Scratch checkpoint mismatch')
        encoder.load_state_dict(state['encoder'])
        decoder.load_state_dict(state['decoder'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['cpu_rng'])
        torch.cuda.set_rng_state(state['cuda_rng'],device=device)
        random.setstate(state['python_rng'])
        del state
    if start in NODES and latest.exists() and not (root/f'step-{start}.pt').exists():
        os.link(latest,root/f'step-{start}.pt')
    observer = None

    @torch.no_grad()
    def evaluate_node(step):
        nonlocal observer
        if step >= 6000 and observer is None:
            observer = prepare_identity_observer(cfg,adapter,clean_report,args.identity_pairs,panels['new'],resources)
        identity_rows = []

        def observe(images, target_indices, rows):
            identity_rows.extend(observer(images,target_indices,rows))

        temp = root / f'validation-{step}.tmp'
        panels_result = evaluate(encoder,decoder,adapter,panels,temp,priors_by_panel=panel_priors,
                                 observe_swap=observe if observer else None)
        temp.replace(root / f'validation-{step}.jsonl')
        record = dict(step=step,panels=panels_result)
        if observer:
            require(len(identity_rows) == 556, 'Identity evaluation did not score every validation pair')
            path = root / f'identity-{step}.tmp'
            with path.open('w',encoding='utf-8') as stream:
                for row in identity_rows:
                    stream.write(json.dumps(row,allow_nan=False)+'\n')
            path.replace(root / f'identity-{step}.jsonl')
            record['identity'] = identity_summary(identity_rows)
        probe = encoder(covers[:8],labels[codes[0,:8]],priors[:8])
        alternate = encoder(covers[:8],1-labels[codes[0,:8]],priors[:8])
        record['branches'] = dict(adv_residual_rms=float(probe.adv_residual.square().mean().sqrt()),
                                  msg_residual_rms=float(probe.msg_residual.square().mean().sqrt()),
                                  adv_gate_mean=float(probe.adv_gate.mean()),msg_gate_mean=float(probe.msg_gate.mean()),
                                  message_image_delta_rms=float((probe.image-alternate.image).square().mean().sqrt()))
        atomic_json(root / f'evaluation-{step}.json',record)
        if step == 6000 and panels_result['new']['direct']['all']['ber'] >= .45:
            print('Direct BER is still near chance at step 6000. Review message response and branch gradients.',
                  flush=True)
        print(json.dumps(dict(step=step,direct=panels_result['new']['direct']['all'],
                              swap=panels_result['new']['swap']['all'],identity=record.get('identity'))),flush=True)

    def save(step, log_offset):
        require(all(int(state['step']) == step for state in optimizer.state.values()), 'Adam step mismatch')
        require(tensor_digest(adapter) == oldpan and all(p.grad is None for p in adapter.parameters()), 'Oldpan changed')
        temp = root / 'latest.tmp'
        torch.save(dict(step=step,data_cursor=16*step,log_offset=log_offset,metadata=meta,
                        encoder=encoder.state_dict(),decoder=decoder.state_dict(),optimizer=optimizer.state_dict(),
                        scheduler=scheduler.state_dict(),cpu_rng=torch.get_rng_state(),
                        cuda_rng=torch.cuda.get_rng_state(device),python_rng=random.getstate()),temp)
        temp.replace(latest)
        if step in NODES:
            os.link(latest,root/f'step-{step}.pt')

    begin = time.monotonic()
    with (root/'train.jsonl').open('a+b') as stream:
        require(stream.tell() >= offset, 'Training log is shorter than the durable cursor')
        stream.truncate(offset)
        stream.seek(offset)
        if not latest.exists():
            evaluate_node(0)
            save(0,0)
        for step in range(start+1,STEPS+1):
            encoder.train()
            decoder.train()
            batch = indices[step-1].tolist()
            ramp = .1+.9*min((step-1)/1999,1)
            row = update(encoder,decoder,adapter,optimizer,parameters,covers,ids,batch,labels[codes[step-1]],step,
                         meta['message_weights'],ofea_weight=weight*ramp,track_ofea=True,priors=priors)
            row.update(lr=optimizer.param_groups[0]['lr'],ofea_weight=weight*ramp)
            if step in (1,100,1000,6000):
                for name in ('adv_head','msg_head'):
                    grads = [p.grad for p in getattr(encoder,name).parameters() if p.grad is not None]
                    norm = sum(float(g.detach().square().sum()) for g in grads)**.5
                    require(np.isfinite(norm) and norm > 0, f'{name} did not receive a finite gradient')
                    row[name+'_grad_norm'] = norm
            scheduler.step()
            stream.write((json.dumps(row,allow_nan=False)+'\n').encode('utf-8'))
            if step in NODES:
                evaluate_node(step)
            if step % 1000 == 0 or step in (1,100):
                stream.flush()
                os.fsync(stream.fileno())
                save(step,stream.tell())
            if step % 50 == 0 or step in (1,100):
                stream.flush()
                rate = (step-start)/(time.monotonic()-begin)
                status = dict(step=step,total_steps=STEPS,updates_per_second=rate,
                              peak_cuda_mib=torch.cuda.max_memory_allocated(device)/2**20)
                atomic_json(root/'progress.json',status)
                print(json.dumps(status),flush=True)
    atomic_json(root/'complete.json',dict(meta,step=STEPS,final_checkpoint_sha256=file_hash(root/f'step-{STEPS}.pt')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','design','baseline','identity-pairs','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--device',required=True)
    parser.add_argument('--arm',choices=('control','ofea'),required=True)
    parser.add_argument('--seed',type=int,default=0)
    with ExitStack() as resources:
        run(parser.parse_args(),resources)
