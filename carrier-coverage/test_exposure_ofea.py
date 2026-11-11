"""Regression checks for the OFEA update on a small differentiable swap model."""

import json
from contextlib import ExitStack
from dataclasses import asdict
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import train_exposure as training
import exposure_identity as identity


class SwapModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.), requires_grad=False)
        self.calls = 0
        self.pairs = []

    def forward_pair(self, image, source_ids, target_ids):
        assert all(source != target for source, target in zip(source_ids, target_ids, strict=True))
        self.calls += 1
        target = self.scale * image
        source = image
        if image.requires_grad:
            target.retain_grad()
        self.pairs.append((target, source))
        return SimpleNamespace(image=source.sigmoid(), target_logits=target, source_logits=source)

    def swap(self, _, image, source_ids, target_ids):
        return self.forward_pair(image, source_ids, target_ids).image


def train_once(weight, *, track_ofea=False):
    encoder = nn.Conv2d(1, 1, 1, bias=False)
    decoder = nn.Sequential(nn.Flatten(), nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        encoder.weight.fill_(.95)
        decoder[1].weight.fill_(.4)
    adapter = SwapModel()
    parameters = list(encoder.parameters()) + list(decoder.parameters())
    optimizer = torch.optim.SGD(parameters, lr=.01)
    covers = torch.linspace(.1, .5, 16).reshape(16, 1, 1, 1)
    ids = ['cage'] * 7 + ['trump'] * 9
    indices = list(range(5)) + list(range(7, 10)) + [5, 6] + list(range(10, 16))
    bits = (torch.arange(16) % 2).float().reshape(16, 1)
    row = training.update(encoder, decoder, adapter, optimizer, parameters, covers, ids, indices,
                          bits, 117001, {'direct': .5, 'swap': .5}, ofea_weight=weight, track_ofea=track_ofea)
    return encoder, decoder, adapter, row


def test_ofea_uses_both_native_features_and_equal_identity_weights(monkeypatch):
    monkeypatch.setattr(training, 'watermarked', lambda model, x, _: (model(x), model(x)))
    monkeypatch.setattr(training, 'image_structure_loss', lambda a, b: (a - b).square().mean())
    baseline_encoder, baseline_decoder, _, baseline = train_once(0.)
    control_encoder, control_decoder, _, control = train_once(0., track_ofea=True)
    torch.testing.assert_close(control_encoder.weight, baseline_encoder.weight, rtol=0, atol=0)
    torch.testing.assert_close(control_decoder[1].weight, baseline_decoder[1].weight, rtol=0, atol=0)
    assert all(control[s]['weighted_ofea'] == 0 and control[s]['ofea'] > 0 for s in ('designated','common'))
    encoder, decoder, adapter, with_ofea = train_once(.25)
    assert adapter.calls == 2
    assert adapter.scale.grad is None
    assert all(t.grad is not None and t.grad.norm() > 0 for t, _ in adapter.pairs)
    assert not torch.equal(encoder.weight, baseline_encoder.weight)
    torch.testing.assert_close(decoder[1].weight, baseline_decoder[1].weight, rtol=0, atol=0)
    per_image = (torch.linspace(.1, .5, 16) * .95).square()
    expected = float(.5 * (per_image[:7].mean() + per_image[7:].mean()))
    logged = sum(with_ofea[k]['ofea'] for k in ('designated', 'common'))
    assert logged == pytest.approx(expected)
    for slot in ('designated', 'common'):
        assert with_ofea[slot]['contribution'] - baseline[slot]['contribution'] == pytest.approx(
            .25 * with_ofea[slot]['ofea'], abs=1e-7)


def test_ofea_cannot_change_the_weighted_recipe():
    with pytest.raises(ValueError, match='50:50'):
        training.validate_ofea_weight(.1, 'swap-weighted')


def test_fork_pins_the_existing_checkpoint_and_uses_new_validation_nodes(tmp_path):
    parent = tmp_path / 'parent'
    parent.mkdir()
    checkpoint = parent / 'step-117000.pt'
    checkpoint.write_bytes(b'checkpoint')
    base = {'plan_sha256':'schedule', 'device':'cuda:1'}
    (parent / 'config.json').write_text(json.dumps(dict(base,device='cuda:3')))
    (parent / 'validation-117000.jsonl').touch()
    (parent / 'evaluation-117000.json').touch()
    args = SimpleNamespace(resume_from=checkpoint, through_step=118000, output=tmp_path/'new',
                           ofea_weight=.01, carriers=416)
    source, parent_config, meta = training.fork_metadata(args,'swap-equal',base)
    assert source == checkpoint.resolve()
    assert parent_config['device'] == 'cuda:3'
    assert meta['fork_checkpoint_sha256'] == training.file_hash(checkpoint)
    assert meta['fork_step'] == 117000 and meta['ofea_weight'] == .01
    assert meta['nodes'] == [117000,117250,117500,118000]
    args.output = parent
    with pytest.raises(ValueError,match='separate output'):
        training.fork_metadata(args,'swap-equal',base)


def test_identity_joint_success_uses_the_same_message_rows(monkeypatch,tmp_path):
    class Evaluator:
        evaluator_hash = 'identity'
        preprocessing_hash = 'preprocess'

        def validate_threshold(self, threshold):
            assert threshold.evaluator_hash == self.evaluator_hash

        def embeddings(self, images):
            return images, images.norm(dim=1) > 0

    monkeypatch.setattr(identity,'make_identity_evaluator',lambda *_args,**_kwargs:Evaluator())
    records = [dict(sample_id=name,identity_id=name,reference_eligible=True) for name in ('cage','trump')]
    pairs = [dict(condition='swap_none',split='val',target_sample_id=target,source_sample_id=source,
                  target_id=target,source_id=source) for target,source in (('cage','trump'),('trump','cage'))]
    path = tmp_path / 'pairs.jsonl'
    path.write_text(''.join(json.dumps(pair)+'\n' for pair in pairs))
    threshold = identity.ThresholdSpec('identity','preprocess','reference',.01,.5,0,2,2)
    report = dict(pair_manifest_hash=training.file_hash(path),threshold=asdict(threshold))
    inputs = torch.eye(2)
    panel = (inputs,records,None,None,inputs.flip(0))
    cfg = {'assets':{'identity_evaluator':{}}}
    with ExitStack() as resources:
        observe = identity.prepare_identity_observer(cfg,SimpleNamespace(checkpoint_hash='swap'),report,path,panel,resources)
        messages = [dict(sample_id=r['sample_id'],target_id=r['identity_id'],group_id=r['sample_id'],
                         message_index=0,message=[0]*15,bit_errors=0) for r in records]
        rows = observe(torch.tensor([[1.,0.],[0.,0.]]),(0,1),messages)
    metrics = identity.identity_summary(rows)['all']
    assert metrics['eligible_count'] == 2 and metrics['and'] == .5
    assert metrics['detection_failure_count'] == 1
