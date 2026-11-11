"""Score identity and watermark recovery on the same protected swap outputs."""

import json

import torch

from rgwm.evaluation.identity import ThresholdSpec
from rgwm.evaluation.protocol import summarize
from rgwm.evaluation.worker_identity import make_identity_evaluator
from rgwm.legacy import file_hash
from run_probe import opposite, require


def prepare_identity_observer(cfg, adapter, report, pairs_path, panel, resources):
    require(report and report.get('pair_manifest_hash') == file_hash(pairs_path),
            'Identity pairs must match the accepted clean-quality report')
    covers, records, _, _, clean = panel
    by_sample = {r['sample_id']:i for i,r in enumerate(records)}
    by_target = {}
    with pairs_path.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip():
                continue
            pair = json.loads(line)
            if pair['condition'] != 'swap_none':
                continue
            target = pair['target_sample_id']
            require(pair['split'] == 'val' and target in by_sample and target not in by_target,
                    'Expected one frozen swap reference per validation carrier')
            source = records[by_sample[pair['source_sample_id']]]
            target_id = records[by_sample[target]]['identity_id']
            require(source['reference_eligible'] is True and pair['target_id'] == target_id and
                    source['identity_id'] == pair['source_id'] == opposite([target_id])[0],
                    'Invalid source reference in frozen validation pairs')
            by_target[target] = pair['source_sample_id']
    require(set(by_target) == set(by_sample), 'Incomplete frozen validation references')
    source_ids = [by_target[r['sample_id']] for r in records]

    evaluator = make_identity_evaluator(cfg['assets']['identity_evaluator'], device='cpu')
    if hasattr(evaluator, 'close'):
        resources.callback(evaluator.close)
    threshold = ThresholdSpec(**report['threshold'])
    evaluator.validate_threshold(threshold)
    require(evaluator.evaluator_hash != adapter.checkpoint_hash, 'Identity evaluator must be independent of Oldpan')

    @torch.inference_mode()
    def batch_embeddings(images):
        parts = [evaluator.embeddings(images[i:i+8]) for i in range(0,len(images),8)]
        return tuple(torch.cat(values).cpu() for values in zip(*parts,strict=True))

    embeddings, valid_refs = batch_embeddings(covers)
    refs = [by_sample[source] for source in source_ids]
    require(bool(valid_refs[refs].all()), 'Frozen source references must be detection-valid')
    references = embeddings[refs]
    clean_embeddings, clean_valid = batch_embeddings(clean)
    clean_scores = (clean_embeddings * references).sum(1)
    eligible = clean_valid & (clean_scores >= threshold.threshold)
    require(bool(eligible.any()), 'No clean swap matches its source identity')

    def observe(images, indices, message_rows):
        features, valid = batch_embeddings(images)
        indices = list(indices)
        scores = (features * references[indices]).sum(1)
        result = []
        for index, row, score, detected in zip(indices,message_rows,scores,valid,strict=True):
            detected = bool(detected)
            result.append(dict(sample_id=row['sample_id'], group_id=row['group_id'],
                               target_id=row['target_id'], source_id=opposite([row['target_id']])[0],
                               source_sample_id=source_ids[index], message_index=row['message_index'],
                               message_length=len(row['message']), bit_errors=row['bit_errors'],
                               clean_eligible=bool(eligible[index]),
                               clean_identity_score=float(clean_scores[index]) if bool(clean_valid[index]) else None,
                               detection_status='valid' if detected else 'failed',
                               identity_score=float(score) if detected else None,
                               D=int(detected and float(score) < threshold.threshold), T=int(row['bit_errors'] == 0)))
        return result

    return observe


def identity_summary(rows):
    return {'all':summarize(rows),
            **{name:summarize(r for r in rows if r['target_id'] == name) for name in ('cage','trump')}}
