"""Summarize a running scratch experiment and replay fixed validation examples."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rgwm.models.factory import build_encoder
from run_joint import EPSILON, MessageDecoder, make_adapter, make_dataset, tensor_digest
from train_exposure import file_hash, load_run_config


ARMS = ("control", "ofea")
WINDOW = 1000


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            yield json.loads(line)


def key(row):
    return row["sample_id"], row["message_index"]


def summarize_train(folder, through):
    names = ("direct_bce", "swap_bce", "ofea")
    rows = []
    acc = dict.fromkeys(names, 0.0)
    for step, row in enumerate(read_rows(folder / "train.jsonl"), start=1):
        assert row["step"] == step
        for name in names:
            acc[name] += row["designated"][name] + row["common"][name]
        if step % WINDOW == 0:
            rows.append(dict(step=step, **{name: total / WINDOW for name, total in acc.items()}))
            acc = dict.fromkeys(names, 0.0)
        if step == through:
            break
    assert rows[-1]["step"] == through
    return rows


def select_examples(swap_rows, identity_rows):
    selected = []
    used = set()
    for direction in ("cage", "trump"):
        ordered = sorted(
            (k for k, row in swap_rows["ofea"].items() if row["target_id"] == direction),
            key=lambda k: (swap_rows["ofea"][k]["bit_errors"], k),
        )
        chosen = ordered[len(ordered) // 2]
        selected.append(("median", chosen))
        used.add(chosen[0])
    eligible = sorted(
        k for k, row in identity_rows["ofea"].items() if row["clean_eligible"] and row["D"]
    )
    for chosen in eligible:
        if chosen[0] not in used:
            selected.append(("ofea_block", chosen))
            used.add(chosen[0])
            break
    for successful, failed in (("control", "ofea"), ("ofea", "control")):
        candidates = sorted(
            (k for k in swap_rows[successful] if k[0] not in used
             and identity_rows[successful][k]["clean_eligible"]
             and all(identity_rows[a][k]["detection_status"] == "valid" for a in ARMS)
             and swap_rows[successful][k]["bit_errors"] == 0
             and swap_rows[failed][k]["bit_errors"] > 0),
            key=lambda k: (swap_rows[failed][k]["bit_errors"], k),
        )
        if candidates:
            selected.append((successful + "_exact", candidates[0]))
            used.add(candidates[0][0])
    return selected


@torch.inference_mode()
def replay(args, folders, step, swap_rows, direct_rows, identity_rows):
    cfg = load_run_config(Path("configs/M.yaml"), Path("reports/author64-watermark-runtime-v1-2threads/assets.yaml").resolve())
    design_path = Path("runs/author64-message-sampling/dynamic-full/design.json")
    design = read_json(design_path)
    data = make_dataset(cfg, split="val")
    assert len(data) == 139 and [row["sample_id"] for row in data.records] == design["val_ids"]
    pairs = [tuple(pair) for pair in design["val_pairs"]]
    positions = {(design["val_ids"][i], j): k for k, (i, j) in enumerate(pairs)}
    assert len(positions) == 556
    device = torch.device(args.device)
    adapter = make_adapter(cfg, device).eval()
    oldpan_hash = tensor_digest(adapter)
    design_hash = file_hash(design_path)
    manifest_hash = file_hash(cfg["assets"]["manifest"])
    models = {}
    for arm, folder in folders.items():
        state = torch.load(folder / f"step-{step}.pt", weights_only=True, map_location="cpu", mmap=True)
        meta = state["metadata"]
        assert state["step"] == step and meta["arm"] == arm
        assert meta["oldpan"] == oldpan_hash and meta["design_sha256"] == design_hash
        assert meta["manifest_sha256"] == manifest_hash and meta["test_images_read"] == 0
        encoder = build_encoder("M", message_length=15, epsilon=EPSILON,
                                head_channels=tuple(cfg["model"]["head_channels"])).to(device).eval()
        decoder = MessageDecoder(15).to(device).eval()
        encoder.load_state_dict(state["encoder"])
        decoder.load_state_dict(state["decoder"])
        models[arm] = encoder, decoder
        del state
    details, images = [], []
    for kind, pair_key in select_examples(swap_rows, identity_rows):
        position = positions[pair_key]
        offset = position // 8 * 8
        k = position - offset
        ii, jj = zip(*pairs[offset:offset + 8])
        items = [data[i] for i in ii]
        covers = torch.stack([item["target"] for item in items]).to(device)
        priors = torch.stack([item["priors"] for item in items]).to(device)
        targets = [data.records[i]["identity_id"] for i in ii]
        sources = ["trump" if target == "cage" else "cage" for target in targets]
        messages = torch.tensor([design["val_messages"][j] for j in jj], dtype=torch.float32, device=device)
        clean = adapter.swap(None, covers, sources, targets)
        record = dict(kind=kind, sample_id=pair_key[0], message_index=pair_key[1],
                      target=targets[k], source=sources[k], message=messages[k].int().tolist())
        frames = [covers[k], clean[k]]
        for arm, (encoder, decoder) in models.items():
            protected = encoder(covers, messages, priors).image
            swapped = adapter.swap(None, protected, sources, targets)
            frames.extend([protected[k], swapped[k]])
            direct, swap = direct_rows[arm][pair_key], swap_rows[arm][pair_key]
            for output, saved in ((protected, direct), (swapped, swap)):
                assert saved["message"] == record["message"]
                bits = (decoder(output)[k] >= 0).int().tolist()
                assert bits == saved["predicted_bits"], (arm, pair_key, saved["path"])
            result = identity_rows[arm][pair_key]
            assert result["bit_errors"] == swap["bit_errors"]
            record[arm] = dict(psnr=direct["psnr"], direct_errors=direct["bit_errors"],
                               swap_errors=swap["bit_errors"], identity_score=result["identity_score"],
                               eligible=result["clean_eligible"], detection=result["detection_status"],
                               blocked=bool(result["D"]))
        frame = torch.stack(frames)
        assert torch.isfinite(frame).all() and frame.min() >= 0 and frame.max() <= 1
        images.append(frame.permute(0, 2, 3, 1).cpu().numpy())
        details.append(record)
    np.savez_compressed(args.output / "examples.npz", images=np.stack(images))
    return details


def run(args):
    folders = {arm: args.run / arm for arm in ARMS}
    progress = {arm: read_json(folder / "progress.json") for arm, folder in folders.items()}
    nodes = {}
    for arm, folder in folders.items():
        records = [read_json(path) for path in folder.glob("evaluation-*.json")]
        nodes[arm] = {r["step"]: r for r in records if (folder / f"step-{r['step']}.pt").exists()}
    common = sorted(set(nodes["control"]) & set(nodes["ofea"]))
    step = common[-1]
    assert step >= 6000
    through = min(row["step"] for row in progress.values()) // WINDOW * WINDOW
    train = {arm: summarize_train(folder, through) for arm, folder in folders.items()}
    swap_rows, direct_rows, identity_rows = {}, {}, {}
    for arm, folder in folders.items():
        relevant = [row for row in read_rows(folder / f"validation-{step}.jsonl") if row["panel"] == "new"]
        swap_rows[arm] = {key(row): row for row in relevant if row["path"] == "swap"}
        direct_rows[arm] = {key(row): row for row in relevant if row["path"] == "direct"}
        identity_rows[arm] = {key(row): row for row in read_rows(folder / f"identity-{step}.jsonl")}
        assert len(swap_rows[arm]) == len(direct_rows[arm]) == len(identity_rows[arm]) == 556
    assert swap_rows["control"].keys() == swap_rows["ofea"].keys()
    assert identity_rows["control"].keys() == identity_rows["ofea"].keys() == swap_rows["ofea"].keys()
    pair_counts = dict(control_only=0, ofea_only=0, both=0, neither=0)
    identity_delta = []
    for k in swap_rows["ofea"]:
        a, b = (swap_rows[arm][k]["bit_errors"] == 0 for arm in ARMS)
        pair_counts[("neither", "ofea_only", "control_only", "both")[2 * a + b]] += 1
        c, o = (identity_rows[arm][k] for arm in ARMS)
        assert c["clean_eligible"] == o["clean_eligible"]
        if c["clean_eligible"] and c["detection_status"] == o["detection_status"] == "valid":
            identity_delta.append(o["identity_score"] - c["identity_score"])
    args.output.mkdir(parents=True, exist_ok=True)
    examples = replay(args, folders, step, swap_rows, direct_rows, identity_rows)
    report = dict(progress=progress, comparison_step=step, train_through=through, window=WINDOW,
                  evaluations={arm: [nodes[arm][s] for s in common] for arm in ARMS},
                  train=train, pair_counts=pair_counts, examples=examples, test_images_read=0,
                  source_similarity_delta=dict(n=len(identity_delta), mean=float(np.mean(identity_delta)),
                                               median=float(np.median(identity_delta))))
    (args.output / "review.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda:2")
    run(parser.parse_args())
