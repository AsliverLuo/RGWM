"""Render convergence and verified face-swap examples from archived runs."""

import argparse
import json
from pathlib import Path

import numpy as np


RECIPES = ('mixed', 'swap-weighted', 'swap-equal')
STYLES = (('混合配方', '#52667b', 'o', '--'),
          ('全量换脸 75:25', '#007d68', 's', '-.'),
          ('全量换脸 50:50', '#9b3874', '^', '-'))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def records(path):
    with path.open(encoding='utf-8') as stream:
        return [json.loads(line) for line in stream]


def pair_key(row):
    return row['sample_id'], row['message_index']


def infer(args):
    import torch
    from diagnose_author64_output import OutputEncoder
    from rgwm.cli.common import make_adapter, make_dataset
    from rgwm.config import load_run_config
    from rgwm.evaluation.clean_quality import validate_clean_quality_report
    from rgwm.legacy import file_hash
    from rgwm.models.decoder import MessageDecoder
    from run_joint import watermarked

    roots, meta = {}, {}
    sources = ([(count, args.coverage_root / f'n{count}-seed0') for count in (16,416)] if args.coverage_root else
               [(None, phase / 'swap-equal-seed0') for phase in (args.parent,args.continuation)])
    for label, root in sources:
        manifest = read_json(root / 'complete.json')
        tag = label if label is not None else manifest['steps']
        roots[tag] = root
        meta[tag] = manifest
    first_step, last_step = sorted(roots)
    raw = {}
    for step, root in roots.items():
        rows = records(root / f"validation-{meta[step]['steps']}.jsonl")
        raw[step] = {path: {pair_key(r): r for r in rows if r['panel'] == 'new' and r['path'] == path}
                     for path in ('swap', 'direct')}
        assert all(len(values) == 556 for values in raw[step].values())
    assert raw[first_step]['swap'].keys() == raw[last_step]['swap'].keys()
    final = raw[last_step]['swap']
    selected = []
    for identity in ('cage', 'trump'):
        by_error = sorted((r for r in final.values() if r['target_id'] == identity),
                          key=lambda r: (r['bit_errors'], pair_key(r)))
        median = by_error[len(by_error)//2]
        recovered = next((r for r in by_error if r['bit_errors'] == 0 and
                          raw[first_step]['swap'][pair_key(r)]['bit_errors'] > 0), None)
        kind = '扩充后全对样本' if args.coverage_root else '续训后全对样本'
        if recovered is None:
            recovered = by_error[0]
            kind = '终点全对样本' if recovered['bit_errors'] == 0 else '终点低误码样本'
        selected.extend((('终点误码中位样本', median), (kind, recovered)))

    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    assert all(file_hash(args.design) == m['design_sha256'] and
               file_hash(cfg['assets']['manifest']) == m['manifest_sha256'] for m in meta.values())
    dataset = make_dataset(cfg, split='val')
    assert len(dataset) == 139 and [r['sample_id'] for r in dataset.records] == design['val_ids']
    ids = [r['identity_id'] for r in dataset.records]
    pairs = [tuple(pair) for pair in design['val_pairs']]
    locations = {(design['val_ids'][i], j): index for index, (i, j) in enumerate(pairs)}
    device = torch.device(args.device)
    adapter = make_adapter(cfg, device).eval()
    validate_clean_quality_report(cfg, adapter)
    assert all(adapter.checkpoint_hash == m['oldpan_sha256'] for m in meta.values())
    with torch.inference_mode():
        models = {}
        for step, root in roots.items():
            ckpt = root / 'checkpoints' / f"step-{meta[step]['steps']}.pt"
            assert file_hash(ckpt) == meta[step]['final_checkpoint_sha256']
            state = torch.load(ckpt, weights_only=True, map_location='cpu')
            encoder, decoder = OutputEncoder('residual').to(device).eval(), MessageDecoder(15).to(device).eval()
            encoder.load_state_dict(state['encoder'], strict=True)
            decoder.load_state_dict(state['decoder'], strict=True)
            models[step] = encoder, decoder
            del state

        panels, details = [], []
        for kind, row in selected:
            key = pair_key(row)
            idx = locations[key]
            start = idx - idx % 8
            i, j = zip(*pairs[start:start+8], strict=True)
            position = idx-start
            targets = [ids[k] for k in i]
            sources = ['trump' if t == 'cage' else 'cage' for t in targets]
            covers = torch.stack([dataset[k]['target'] for k in i]).to(device)
            messages = torch.tensor([design['val_messages'][k] for k in j], device=device, dtype=torch.float32)
            assert messages[position].to(torch.int64).tolist() == row['message']
            clean = adapter.swap(None, covers, sources, targets)
            images = {}
            for step, (encoder, decoder) in models.items():
                protected, _ = watermarked(encoder, covers, messages)
                swapped = adapter.swap(None, protected, sources, targets)
                for path, value in (('direct', protected), ('swap', swapped)):
                    predicted = (decoder(value)[position] >= 0).to(torch.int64).tolist()
                    assert predicted == raw[step][path][key]['predicted_bits'], (step, path, key)
                images[step] = protected[position], swapped[position]
            frame = torch.stack((covers[position], images[last_step][0], clean[position],
                                 images[first_step][1], images[last_step][1]))
            assert torch.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 1
            panels.append(frame.permute(0, 2, 3, 1).cpu().numpy())
            details.append(dict(kind=kind, sample_id=key[0], message_index=key[1], direction=targets[position]+' → '+sources[position],
                                psnr=raw[last_step]['direct'][key]['psnr'], direct_errors=raw[last_step]['direct'][key]['bit_errors'],
                                **{f'step{step}_errors': raw[step]['swap'][key]['bit_errors'] for step in roots}))
    np.savez_compressed(args.output / 'examples-raw.npz', images=np.stack(panels))
    (args.output / 'examples-audit.json').write_text(json.dumps(dict(validation_pairs=556, test_images_read=0,
        checkpoint_sha256={step:m['final_checkpoint_sha256'] for step,m in meta.items()},
        selection=f'Upper median bit error by direction and first exact recovery for {last_step} that failed for {first_step}, with best-error fallback',
        comparison='carrier counts at step 6000' if args.coverage_root else 'training steps',
        prediction_replays=4*2*2, examples=details), ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def draw(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from matplotlib.ticker import MaxNLocator

    for font in args.font_file:
        font_manager.fontManager.addfont(font)
    family = font_manager.FontProperties(fname=args.font_file[0]).get_name() if args.font_file else 'Noto Sans CJK SC'
    plt.rcParams.update({'font.family': family, 'axes.unicode_minus': False,
                         'font.size': 10, 'axes.labelcolor': '#344253', 'xtick.color': '#344253',
                         'ytick.color': '#344253', 'text.color': '#162334'})
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.3))
    fig.subplots_adjust(left=.098, right=.969, top=.79, bottom=.128, hspace=.57, wspace=.28)
    coverage = bool(args.coverage_root)
    title = '载体规模与换脸后消息恢复' if coverage else '真实换脸监督的训练与验证曲线'
    subtitle = ('三种训练载体规模  ·  直接与换脸 BCE 权重 75:25  ·  固定验证配对' if coverage else
                '16 张训练载体  ·  三种消息配方  ·  固定 Oldpan 64 与验证配对')
    fig.text(.098, .953, title, fontsize=19, weight='bold')
    fig.text(.098, .91, subtitle, color='#52667b', fontsize=10)
    handles = []
    phases = ((args.coverage_root,) if coverage else
              ((args.initial, args.parent, args.continuation) if args.initial else (args.parent, args.continuation)))
    recipes = ('n16','n64','n416') if coverage else RECIPES
    styles = [(f'{r[1:]} 张', *s[1:]) for r,s in zip(recipes, STYLES, strict=True)] if coverage else STYLES
    maxima = [0., 0., 0., 0.]
    min_ber = 36.
    for recipe, (label, color, marker, style) in zip(recipes, styles, strict=True):
        folders = [root / f'{recipe}-seed0' for root in phases]
        t = [r for root in folders for r in records(root / 'train.jsonl')]
        end = t[-1]['step']
        assert [r['step'] for r in t] == list(range(1,end+1))
        evaluations = [read_json(root / 'evaluation.json') for root in folders]
        assert all(a[-1] == b[0] for a,b in zip(evaluations,evaluations[1:]))
        boundaries = [v[0]['step'] for v in evaluations[1:]]
        nodes = evaluations[0] + [n for v in evaluations[1:] for n in v[1:]]
        train_values = np.array([r['designated']['processed_bce'] if recipe == 'mixed' else
                                 (r['designated']['processed_bce']+r['common']['processed_bce'])/2 for r in t])
        series = (train_values.reshape(-1,100).mean(1),
                  [100*n['panels']['same']['swap']['all']['exact']/512 for n in nodes],
                  [n['panels']['new']['swap']['all']['exact'] for n in nodes],
                  [100*n['panels']['new']['swap']['all']['ber'] for n in nodes])
        for k, (ax, values) in enumerate(zip(axes.flat, series, strict=True)):
            maxima[k] = max(maxima[k], max(values))
            if k == 3:
                min_ber = min(min_ber,min(values))
            x = range(100,end+1,100) if k == 0 else [n['step'] for n in nodes]
            line, = ax.plot(x, values, label=label, c=color, marker=marker, ls=style, lw=1.8, ms=4.5)
            if k == 0:
                handles.append(line)
    for k, (ax, title, y, limits) in enumerate(zip(axes.flat,
            ('A  训练时换脸 BCE','B  同图新消息 · 512 对','C  新图新消息 · 556 对','D  新图新消息 · 556 对'),
            ('每 100 步均值', '换脸后整串恢复率 (%)','换脸后整串恢复数 (条)','换脸后比特错误率 (%)'),
            ((0,max(.65, maxima[0]+.02)),(0,min(100,max(45,10*np.ceil(maxima[1]/10)))),(0,max(10,maxima[2]+1)),
             ((np.floor(min_ber)-1) if coverage else min(26,np.floor(min_ber)-1),max(36,np.ceil(maxima[3]+.05)))), strict=True)):
        ax.set(title=title, xlim=(-.01*end,1.02*end), ylim=limits, ylabel=y, xticks=np.linspace(0,end,5,dtype=int))
        ax.title.set_position((0,.99)); ax.title.set_ha('left'); ax.title.set_weight('bold')
        for step in boundaries:
            ax.axvline(step, c='#9aa9b7', lw=1, ls=':')
        ax.axvspan(end-(500 if coverage else 250),end, color='#e4eaf0', alpha=.55, zorder=0)
        ax.grid(axis='y', c='#d7dee4', lw=.65)
        ax.spines[['top','right']].set_visible(False)
        if k >= 2:
            ax.set_xlabel('优化器更新步')
        if k == 2:
            ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=7))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.095,.89), ncol=3, frameon=False)
    marks = '、'.join(map(str,boundaries))
    note = ('浅色区为最后 500 步。B 使用各组共有的原 16 张图和新消息，C 与 D 使用 139 张未见图和新消息。单种子结果。' if coverage else
            f'虚线标记第 {marks} 步恢复点，浅色区为最后 250 步。验证点来自保存的检查点，曲线为单种子结果。')
    fig.text(.098, .035, note, fontsize=9, color='#52667b')
    fig.savefig(args.output / 'convergence.png', dpi=165, facecolor='white')
    plt.close(fig)

    audit = read_json(args.output / 'examples-audit.json')
    first_step, last_step = sorted(map(int,audit['checkpoint_sha256']))
    with np.load(args.output / 'examples-raw.npz') as archive:
        frames = archive['images']
    fig, axes = plt.subplots(4, 5, figsize=(11.1, 9.2))
    fig.subplots_adjust(left=.174, right=.97, top=.84, bottom=.077, wspace=.09, hspace=.34)
    title = '16 张与 416 张 · 第 6000 步新图换脸实例' if coverage else '50:50 等权组 · 新图换脸实例'
    fig.text(.174, .955, title, fontsize=20, weight='bold')
    fig.text(.174, .916, '固定验证配对  ·  Oldpan 64  ·  原始载体与输出均按 RGB 160×160 显示', color='#52667b', fontsize=10)
    first = f'N{first_step}' if coverage else f'第 {first_step} 步'
    last = f'N{last_step}' if coverage else f'第 {last_step} 步'
    for col, label in enumerate(('原始载体', f'{last} 水印图', '无水印换脸',
                                f'{first} 换脸', f'{last} 换脸')):
        axes[0,col].set_title(label, fontsize=10, weight='bold', pad=10)
    for i, info in enumerate(audit['examples']):
        center = axes[i,0].get_position().y0 + axes[i,0].get_position().height/2
        direction = info['direction'].replace('cage','Cage').replace('trump','Trump')
        fig.text(.153, center+.016, direction, ha='right', va='center', fontsize=10, weight='bold')
        fig.text(.153, center-.014, info['kind'], ha='right', va='center', fontsize=9, color='#52667b')
        for j, ax in enumerate(axes[i]):
            ax.imshow(frames[i,j], interpolation='nearest')
            ax.set_axis_off()
        psnr = info['psnr']
        axes[i,1].text(.5,-.095,f'{psnr:.2f} dB · 直接错 {info["direct_errors"]}/15 位', transform=axes[i,1].transAxes,
                       fontsize=8.5, ha='center', va='top', color='#52667b')
        for col, step in ((3,first_step),(4,last_step)):
            errors = info[f'step{step}_errors']
            axes[i,col].text(.5,-.095,f'错 {errors}/15 位', transform=axes[i,col].transAxes,
                             fontsize=9, ha='center', va='top', color='#007d68' if errors == 0 else '#52667b',
                             weight='bold' if errors == 0 else 'normal')
    fig.text(.174, .027, '每个方向展示一个终点误码中位样本及一个全对或低误码样本。整体恢复率见曲线。',
             fontsize=9, color='#52667b')
    fig.savefig(args.output / 'face-swap-examples.png', dpi=165, facecolor='white')
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('infer', 'draw'))
    parser.add_argument('--output', required=True, type=Path)
    for name in ('parent','continuation','coverage-root','config','assets','design'):
        parser.add_argument('--' + name, type=Path)
    parser.add_argument('--initial', type=Path, help='First training phase for a three-phase curve')
    parser.add_argument('--font-file', type=Path, action='append', default=[])
    parser.add_argument('--device', default='cuda:2')
    args = parser.parse_args()
    if not args.coverage_root and not (args.parent and args.continuation):
        parser.error('Supply --coverage-root or both --parent and --continuation')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.mode == 'infer':
        infer(args)
    else:
        draw(args)
