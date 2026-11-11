"""Build the frozen shared plan for the N16/N64/N416 coverage experiment."""

import argparse
from collections import Counter, deque
from pathlib import Path
import random

import diagnose_author64_output as protocol
from rgwm.cli.common import make_dataset
from rgwm.config import load_run_config
from rgwm.data.messages import split_message
from rgwm.legacy import file_hash
from run_probe import read_json, require, write_json


SIZES = (16, 64, 416)
IDENTITIES = ('cage', 'trump')
COVERAGE_NODES = (0, 250, 500, 1000, 2000, 3000, 4000, 5000, 5500, 6000)


def select_pools(train, seed):
    protocol.SEED = 0
    original = protocol.chosen_indices(train)
    remaining = [i for i in range(len(train)) if i not in original]
    medium = original.copy()
    rng = random.Random(20261112+seed)
    for identity in IDENTITIES:
        candidates = [i for i in remaining if train.records[i]['identity_id'] == identity]
        rng.shuffle(candidates)
        seen_groups = {train.records[i]['group_id'] for i in original if train.records[i]['identity_id'] == identity}
        distinct, extra = [], []
        for i in candidates:
            group = train.records[i]['group_id']
            (extra if group in seen_groups else distinct).append(i)
            seen_groups.add(group)
        medium.extend((distinct+extra)[:24])
    pools = {16: original, 64: medium, 416: original+remaining}
    for size,pool in pools.items():
        expected = {'cage': 190, 'trump': 226} if size == 416 else dict.fromkeys(IDENTITIES, size//2)
        require(len(set(pool)) == size and Counter(train.records[i]['identity_id'] for i in pool) == expected,
                f'Unexpected carrier pool N{size}')
    return pools


def balanced_batches(pool, records, seed):
    by_identity = [[i for i, raw in enumerate(pool) if records[raw]['identity_id'] == identity]
                   for identity in IDENTITIES]
    queues = [deque(),deque()]
    rng = random.Random(seed)
    while True:
        halves = []
        for indices, queue in zip(by_identity, queues, strict=True):
            current = []
            while len(current) < 8:
                if not queue:
                    fresh = indices.copy()
                    rng.shuffle(fresh)
                    # A cycle can end inside a batch. Defer that batch's old indices.
                    fresh.sort(key=lambda i: i in current)
                    queue.extend(fresh)
                current.append(queue.popleft())
            halves.append(current)
        yield halves[0][:4]+halves[1][:4]+halves[0][4:]+halves[1][4:]


def message_stream(anchor, count, rng):
    anchor = set(map(tuple, anchor))
    pairs = []
    for value in range(1 << 14):
        bits = tuple(map(int,f'{value:015b}'))
        other = tuple(1-b for b in bits)
        if bits not in anchor and other not in anchor and split_message(bits) == split_message(other) == 'train':
            pairs.append((bits,other))
    require(len(pairs) == 10470, 'Expected the frozen training message pool')
    bank = []
    while len(bank) < count:
        candidates = pairs.copy()
        rng.shuffle(candidates)
        carry = len(bank) % 16
        if carry:
            recent = set(bank[-carry:])
            candidates.sort(key=lambda pair: pair[0] in recent)
        bank.extend(message for pair in candidates for message in pair)
    return bank[:count]


def assign_messages(messages, batches, used, rng):
    choices = []
    for bits in messages:
        slots = [i for i in range(16) if all((batch[i],bits) not in used[size] for size,batch in batches.items())]
        rng.shuffle(slots)
        choices.append(slots)
    owners = [-1]*16

    def place(label, visited):
        for slot in choices[label]:
            if slot in visited:
                continue
            visited.add(slot)
            if owners[slot] == -1 or place(owners[slot], visited):
                owners[slot] = label
                return True
        return False

    for label in sorted(range(16), key=lambda i: len(choices[i])):
        require(place(label,set()), 'No shared message assignment avoids repeated carrier-message pairs')
    ordered = [messages[i] for i in owners]
    for size,batch in batches.items():
        used[size].update(zip(batch,ordered,strict=True))
    return ordered


def create_plan(args):
    cfg = load_run_config(args.config, args.assets.resolve())
    design = read_json(args.design)
    manifest_hash = file_hash(cfg['assets']['manifest'])
    require(manifest_hash == design['manifest_sha256'], 'Frozen manifest mismatch')
    train = make_dataset(cfg, split='train')
    require(len(train) == 416 and not train.augment, 'Unexpected frozen train split')
    pools = select_pools(train, args.seed)
    streams = {n: balanced_batches(pool,train.records,20261113+args.seed+n) for n,pool in pools.items()}
    rng = random.Random(20261114+args.seed)
    bank = message_stream(design['train_messages'], 16*args.steps, rng)
    used = {n:set() for n in pools}
    batches = []
    for i in range(args.steps):
        indices = {size: next(stream) for size,stream in streams.items()}
        messages = bank[16*i:16*(i+1)]
        require(len(set(messages)) == 16 and all(len(set(batch)) == 16 for batch in indices.values()), 'Duplicate within a batch')
        batches.append(dict(indices=indices,message=assign_messages(messages,indices,used,rng)))
    require(all(len(pairs) == 16*args.steps for pairs in used.values()), 'Repeated carrier-message pair')
    pool_ids = {size:[train.records[i]['sample_id'] for i in pool] for size,pool in pools.items()}
    frequencies = dict(Counter(Counter(bank).values()))
    require(not args.output.exists(), 'Shared plan already exists')
    write_json(args.output, dict(seed=args.seed,steps=args.steps,manifest_sha256=manifest_hash,design_sha256=file_hash(args.design),
               pools=pool_ids,same_messages=protocol.balanced_messages('val',32,20261109),batches=batches,
               message_frequency_histogram=frequencies))
    print(f'Frozen {args.steps} updates across {SIZES}, message frequency histogram {frequencies}',flush=True)


def group_plan(shared, size, train):
    ids = shared['pools'][str(size)]
    lookup = {r['sample_id']:i for i,r in enumerate(train.records)}
    require(len(set(ids)) == size and ids[:16] == shared['pools']['16'], 'Unexpected nested training pool')
    selected = [lookup[i] for i in ids]
    batches = []
    for batch in shared['batches']:
        indices, messages = batch['indices'][str(size)], batch['message']
        micros = [dict(slot=slot,indices=indices[start:start+8],message=messages[start:start+8])
                  for slot,start in (('designated',0),('common',8))]
        for micro in micros:
            require(Counter(train.records[selected[i]]['identity_id'] for i in micro['indices']) == {'cage':4,'trump':4},
                    'Imbalanced coverage microbatch')
        batches.append(dict(micros=micros))
    return selected,dict(train_ids=ids,same_messages=shared['same_messages'],batches=batches)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config','assets','design','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=6000)
    parser.add_argument('--seed',type=int,default=0)
    create_plan(parser.parse_args())
