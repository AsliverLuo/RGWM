"""Repeat the fixed-pair fit diagnostic with balanced held-out carriers."""

import argparse
from collections import Counter
import json
from pathlib import Path
import random

import torch

from run_probe import metrics, read_json, require, tensor_digest, training_images, write_json
from diagnose_author64_output import OutputEncoder
from rgwm.cli.common import make_adapter, make_dataset
from rgwm.config import load_run_config
from rgwm.legacy import file_hash
from rgwm.models.decoder import MessageDecoder


@torch.inference_mode()
def run(args):
    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    train, val = (make_dataset(cfg, split=split) for split in ('train', 'val'))
    require(len(train) == 416 and len(val) == 139, 'Unexpected data split')
    device = torch.device(args.device)
    lookup = {row['sample_id']: i for i, row in enumerate(train.records)}
    rng = random.Random(20261109)
    heldout = {identity: rng.sample([i for i, r in enumerate(val.records)
                                     if r['identity_id'] == identity], 32)
               for identity in ('cage', 'trump')}
    adapter = make_adapter(cfg, device).eval()
    encoder = OutputEncoder('residual').to(device).eval()
    base = torch.load(args.checkpoint, weights_only=True, map_location='cpu')
    encoder.load_state_dict(base['encoder'], strict=True)
    del base
    result = {}
    output_rows = []
    for directory in args.run:
        metadata = read_json(directory / 'complete.json')
        require(metadata['frozen_before'] == {'encoder': tensor_digest(encoder), 'oldpan': tensor_digest(adapter)},
                'Frozen model mismatch')
        require(file_hash(args.design) == metadata['design_sha256'] and
                file_hash(cfg['assets']['manifest']) == metadata['manifest_sha256'], 'Frozen data mismatch')
        plan = read_json(directory / 'training_plan.json')
        decoder = MessageDecoder(15).to(device).eval()
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            checkpoint = torch.load(directory / 'checkpoints' / 'step-1000.pt', weights_only=True, map_location='cpu')
        require(checkpoint['step'] == 1000 and checkpoint['data_cursor'] == 16000 and
                bool(checkpoint['optimizer']['state']), 'Incomplete training state')
        decoder.load_state_dict(checkpoint['decoder'], strict=True)
        del checkpoint
        require(tensor_digest(decoder) == metadata['decoder_final'], 'Decoder checkpoint mismatch')
        indices = [lookup[plan['ordered_train_ids'][i]] for batch in plan['batches'][:4] for i in batch['covers']]
        old = [m for batch in plan['batches'][:4] for m in batch['messages']]
        ids = [train.records[i]['identity_id'] for i in indices]
        require(Counter(ids) == {'cage': 32, 'trump': 32}, 'Unbalanced training panel')
        by_identity = {identity: iter(indices) for identity, indices in heldout.items()}
        balanced_new = [next(by_identity[identity]) for identity in ids]
        panels = (
            ('seen_pair', train, indices, old),
            ('seen_cover_new_message', train, indices,
             [design['val_messages'][i % len(design['val_messages'])] for i in range(64)]),
            ('new_cover_seen_message', val, balanced_new, old),
        )
        result[directory.name] = {}
        for name, dataset, cover_indices, messages in panels:
            covers = torch.stack([dataset[i]['target'] for i in cover_indices]).to(device)
            labels = torch.tensor(messages, dtype=torch.float32, device=device)
            errors = []
            for start in range(0, 64, 8):
                bits = labels[start:start+8]
                output = training_images(encoder, adapter, covers[start:start+8], bits, ids[start:start+8], 'd2', 0)
                pred = (decoder(output) >= 0).to(torch.int64)
                batch_errors = (pred != bits).sum(dim=1).tolist()
                errors.extend(batch_errors)
                for j, prediction in enumerate(pred.tolist(), start=start):
                    output_rows.append(dict(run=directory.name, panel=name, target_id=ids[j],
                                            sample_id=dataset.records[cover_indices[j]]['sample_id'],
                                            message=messages[j], predicted_bits=prediction,
                                            bit_errors=batch_errors[j-start]))
            result[directory.name][name] = {'all': metrics(errors)}
            for identity in ('cage', 'trump'):
                result[directory.name][name][identity] = metrics([e for e, i in zip(errors, ids) if i == identity])
        require(result[directory.name]['seen_pair']['all'] == metadata['fit_diagnostic']['seen_cover_seen_pair'],
                'Fixed training pair result did not replay')
        print(directory.name, [(panel, value['all']) for panel, value in result[directory.name].items()])
    write_json(args.output / 'fit_panels.json', result)
    with (args.output / 'fit_predictions.jsonl').open('w', encoding='utf-8') as stream:
        for row in output_rows:
            stream.write(json.dumps(row) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assets', 'design', 'checkpoint', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--run', type=Path, nargs='+', required=True)
    parser.add_argument('--device', default='cuda:4')
    run(parser.parse_args())
