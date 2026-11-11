"""Compare 50:50 and 75:25 at matched updates on identical validation pairs."""

import argparse
import json
from pathlib import Path

from analyze_joint import PATHS, contrasts, validation
from analyze_probe import load, sha256


def metrics(entry):
    panels = entry['panels']
    new = panels['new']
    return dict(swap=new['swap'], same_swap=panels['same']['swap']['all'],
                direct=new['direct']['all'], png=new['png']['all'], jpeg50=new['jpeg50']['all'],
                quality=new['quality'])


def compare_node(step, weighted_dir, equal_dir, weighted_node, equal_node):
    inputs = zip((weighted_dir,equal_dir),(weighted_node,equal_node),strict=True)
    weighted, equal = [validation(folder,step,node['panels'],dict.fromkeys(PATHS,PATHS['same']))
                       for folder,node in inputs]
    return dict(step=step, weighted=metrics(weighted_node), equal=metrics(equal_node),
                new_swap_delta=contrasts(weighted['new']['swap'], equal['new']['swap']),
                same_swap_delta=contrasts(weighted['same']['swap'], equal['same']['swap']),
                new_direct_delta=contrasts(weighted['new']['direct'], equal['new']['direct']))


def coverage_nodes(weighted_dir, equal_dir):
    weighted, equal = [load(d / 'complete.json') for d in (weighted_dir,equal_dir)]
    assert (weighted['message_recipe'],equal['message_recipe']) == ('swap-weighted','swap-equal')
    assert all(weighted[k] == equal[k] for k in ('checkpoint_sha256','plan_sha256','coverage_plan_sha256',
               'seed','steps','manifest_sha256','design_sha256','oldpan_sha256','start_hashes'))
    assert sha256(weighted_dir/'validation-0000.jsonl') == sha256(equal_dir/'validation-0000.jsonl')
    first, second = [load(d/'evaluation.json') for d in (weighted_dir,equal_dir)]
    assert [v['step'] for v in first] == [v['step'] for v in second]
    return [compare_node(a['step'], weighted_dir, equal_dir, a, b) for a,b in zip(first,second,strict=True)]


def exposure_nodes(weighted_dir, equal_dir, size):
    if not (equal_dir/'config.json').exists():
        return []
    a,b = [load(d / 'config.json') for d in (weighted_dir,equal_dir)]
    assert a['message_weights'] == {'direct':.75,'swap':.25}
    assert b['message_weights'] == {'direct':.5,'swap':.5}
    assert all(a[k] == b[k] for k in ('plan_sha256','carriers','manifest_sha256','design_sha256',
                                      'oldpan_sha256','identity_weights','nodes','test_images_read'))
    result = []
    for step in b['nodes']:
        if step == 6000:
            continue
        files = [d/f'evaluation-{step}.json' for d in (weighted_dir,equal_dir)]
        if all(p.exists() for p in files):
            node = compare_node(step, weighted_dir, equal_dir, *map(load,files))
            result.append(dict(node, exposures=step*16//size))
    return result


def main(args):
    result = {}
    for size in (16,64,416):
        arm = f'n{size}-seed0'
        result[arm] = dict(coverage=coverage_nodes(args.weighted_coverage/arm,args.equal/arm),
                           exposure=exposure_nodes(args.weighted_exposure/arm,args.equal/'exposure'/arm,size))
        for node in [result[arm]['coverage'][-1],*result[arm]['exposure']]:
            diff = node['new_swap_delta']['all']
            print(arm, node['step'], 'exact weighted/equal', node['weighted']['swap']['all']['exact'],
                  node['equal']['swap']['all']['exact'], 'paired WMR delta/CI', diff['wmr_delta'], diff['wmr_ci95'],
                  flush=True)
    path = args.equal/'comparison-50-vs-75.json'
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('weighted-coverage','weighted-exposure','equal'):
        parser.add_argument('--'+key, type=Path,required=True)
    main(parser.parse_args())
