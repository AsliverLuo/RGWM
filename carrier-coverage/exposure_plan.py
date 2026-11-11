"""Freeze a continuation that matches cumulative carrier exposures."""

import argparse
from collections import Counter, defaultdict, deque
from pathlib import Path
import random

import numpy as np

from carrier_coverage import message_stream
from rgwm.cli.common import make_dataset
from rgwm.config import load_run_config
from rgwm.legacy import file_hash
from run_probe import read_json, require, write_json


START = 6000
MILESTONES = {64: (12000, 18000, 24000), 416: (13000, 26000, 39000, 78000, 117000, 156000)}
FIRST_STOP = {64: 24000, 416: 39000}
SEED = 20261116


def message_id(bits):
    return int(''.join(map(str, bits)), 2)


def prior_history(directory, covers):
    meta = read_json(directory / 'complete.json')
    path = directory / 'plan.json'
    require(meta['steps'] == START and file_hash(path) == meta['plan_sha256'], 'Parent plan mismatch')
    plan = read_json(path)
    seen = np.zeros((len(covers), 1 << 15), dtype=np.bool_)
    counts = Counter()
    for batch in plan['batches']:
        for micro in batch['micros']:
            for index, bits in zip(micro['indices'], micro['message'], strict=True):
                code = message_id(bits)
                require(not seen[index, code], 'Repeated parent carrier-message pair')
                seen[index, code] = True
                counts[code] += 1
    require(int(seen.sum()) == START*16, 'Parent exposure count mismatch')
    return plan, seen, counts


def pool_stream(indices, prior, target, rng):
    low = min(prior[i] for i in indices)
    queue = deque()

    def draw(count):
        nonlocal low
        batch = []
        while len(batch) < count:
            if not queue:
                require(low < target, 'Identity quota exhausted')
                fresh = [i for i in indices if prior[i] <= low]
                rng.shuffle(fresh)
                fresh.sort(key=lambda i: i in batch)
                queue.extend(fresh)
                low += 1
            batch.append(queue.popleft())
        require(len(set(batch)) == count, 'Duplicate carrier in one update')
        return batch

    return draw


def carrier_schedule(identities, prior, size):
    rng = random.Random(SEED + size)
    identity_indices = [[i for i,x in enumerate(identities) if x == name] for name in ('cage', 'trump')]
    rows = np.empty((MILESTONES[size][-1]-START, 16), dtype=np.uint16)
    prev = START
    exposure = prior.copy()
    for end in MILESTONES[size]:
        target = 16*end//size
        total = end-prev
        draws = [pool_stream(indices, exposure, target, rng) for indices in identity_indices]
        cage_quota = sum(target-exposure[i] for i in identity_indices[0])
        for offset in range(total):
            c = (offset+1)*cage_quota//total - offset*cage_quota//total
            cage, trump = draws[0](c), draws[1](16-c)
            half = (c+offset%2)//2
            first, second = cage[:half]+trump[:8-half], cage[half:]+trump[8-half:]
            rng.shuffle(first)
            rng.shuffle(second)
            rows[prev-START+offset] = first+second
        segment = rows[prev-START:end-START]
        exposure += np.bincount(segment.ravel(), minlength=size)
        require(np.all(exposure == target), f'N{size} quota mismatch at {end}')
        prev = end
    return rows


def message_batches(base_codes, history, steps, rng):
    """Use lower-frequency complementary pairs before recycling a message."""
    complement = (1 << 15)-1
    tiers = defaultdict(list)
    for code in base_codes:
        require(history[code] == history[code ^ complement], 'Historical complement frequencies differ')
        tiers[history[code]].append(code)
    queue, carry = deque(), []
    for _ in range(steps):
        while len(carry) < 8:
            if not queue:
                low = min(tiers)
                fresh = tiers.pop(low)
                rng.shuffle(fresh)
                fresh.sort(key=lambda code: code in carry)
                queue.extend(fresh)
                tiers[low+1].extend(fresh)
            carry.append(queue.popleft())
        require(len(set(carry)) == 8, 'Duplicate message in one update')
        yield np.asarray([value for code in carry for value in (code, code ^ complement)], dtype=np.uint16)
        carry.clear()


def assign_messages(codes, rows, used, rng):
    """Match labels to positions with no historical pair reused in either run."""
    possible = np.ones((16,16), dtype=np.bool_)
    for size, carriers in rows.items():
        possible &= ~used[size][carriers[:, None], codes[None, :]]
    choices = [np.flatnonzero(available).tolist() for available in possible.T]
    for slots in choices:
        rng.shuffle(slots)
    owners = [-1]*16

    def place(label, visited):
        for position in choices[label]:
            if position in visited:
                continue
            visited.add(position)
            if owners[position] == -1 or place(owners[position], visited):
                owners[position] = label
                return True
        return False

    for label in sorted(range(16), key=lambda i: len(choices[i])):
        require(place(label, set()), 'No shared message matching for this seed')
    ordered = codes[owners]
    for size, carriers in rows.items():
        used[size][carriers, ordered] = True
    return ordered


def main(args):
    cfg = load_run_config(args.config, args.assets.resolve())
    manifest_hash = file_hash(cfg['assets']['manifest'])
    design = read_json(args.design)
    require(design['manifest_sha256'] == manifest_hash, 'Manifest mismatch')
    train = make_dataset(cfg, split='train')
    by_id = {r['sample_id']:r['identity_id'] for r in train.records}
    require(len(by_id) == 416 and not train.augment, 'Train split mismatch')
    rng = random.Random(SEED)
    schedules, seen, histories, parents, pools = {}, {}, {}, {}, {}
    for size in MILESTONES:
        parent = args.parent / f'n{size}-seed0'
        plan, seen[size], histories[size] = prior_history(parent, range(size))
        pools[size] = plan['train_ids']
        ids = [by_id[i] for i in pools[size]]
        require(len(set(pools[size])) == size and Counter(ids) ==
                ({'cage':32, 'trump':32} if size == 64 else {'cage':190, 'trump':226}), 'Carrier pool mismatch')
        schedules[size] = carrier_schedule(ids, seen[size].sum(axis=1), size)
        parents[size] = file_hash(parent / 'plan.json')
    require(histories[64] == histories[416], 'Parent training message streams differ')
    require(set(pools[64]) < set(pools[416]), 'Carrier pools are not nested')
    bank = message_stream(design['train_messages'], 20940, rng)
    bank_ids = set(map(message_id, bank))
    require(set(histories[64]) == bank_ids, 'Frozen training message pool mismatch')
    length = len(schedules[416])
    messages = np.empty((length,16), dtype=np.uint16)
    for row, codes in enumerate(message_batches(sorted(i for i in bank_ids if i < (1 << 14)),
                                                 histories[64], length, rng)):
        active = {size: indices[row] for size,indices in schedules.items() if row < len(indices)}
        messages[row] = assign_messages(codes, active, seen, rng)
        if (row+1) % 30000 == 0:
            print(f'Planned {row+1} continuation updates', flush=True)
    for size, carriers in schedules.items():
        n = len(carriers)
        require(np.all(seen[size].sum(axis=1) == 6000), 'Pair uniqueness or final exposure audit failed')
        frequencies = np.bincount(messages[:n].ravel(), minlength=1 << 15)
        for code, uses in histories[size].items():
            frequencies[code] += uses
        require(int(frequencies.sum()) == (START+n)*16 and
                set(np.flatnonzero(frequencies)) == bank_ids and
                int(frequencies[frequencies > 0].max()-frequencies[frequencies > 0].min()) <= 1,
                'Message frequency mismatch')
        path = args.output / f'n{size}-schedule.npz'
        require(not path.exists(), f'Schedule already exists at {path}')
        np.savez_compressed(path, carriers=carriers, messages=messages[:n])
        metadata = dict(size=size, start=START, nodes=MILESTONES[size], seed=SEED,
                        plan_sha256=file_hash(path), parent_plan_sha256=parents[size],
                        manifest_sha256=manifest_hash, design_sha256=file_hash(args.design),
                        train_ids=pools[size], end_exposures=6000, unique_pairs=(START+n)*16,
                        message_histogram=dict(Counter(map(int, frequencies[frequencies > 0]))))
        write_json(args.output / f'n{size}-schedule.json', metadata)
        print(f'N{size} schedule ready, {n} continuation updates, 6000 exposures per image', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'assets', 'design', 'parent', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    main(parser.parse_args())
