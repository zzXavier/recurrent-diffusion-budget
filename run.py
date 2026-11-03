"""Reproduce and independently evaluate shared vs unshared Looped-DiT on UCI digits."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from src.looped_dit.diffusion import euler_sample, training_loss
from src.looped_dit.model import LoopedMMDiT

ROOT = Path(__file__).resolve().parent
CONFIG = json.loads((ROOT / "configs/experiment.json").read_text())
VARIANTS = {"shared-1": (1, True), "shared-2": (2, True), "shared-4": (4, True),
            "unshared-2": (2, False), "unshared-4": (4, False)}


def build(loops, shared):
    return LoopedMMDiT(image_size=8, patch_size=2, hidden_size=32, num_heads=2,
                       head_dim=16, pca_channels=12, text_dim=10, text_preamble_depth=0,
                       mlp_ratio=1.5, loop_split=(1, 1, 1), num_loops=loops,
                       share_loop_weights=shared)


def data():
    d = load_digits()
    images = torch.tensor(d.images / 8.0 - 1, dtype=torch.float32).unsqueeze(1).repeat(1, 3, 1, 1)
    text = F.one_hot(torch.tensor(d.target, dtype=torch.long), num_classes=10).float().unsqueeze(1)
    train, test = train_test_split(np.arange(len(text)), test_size=CONFIG["validation_fraction"],
                                  random_state=CONFIG["seed"], stratify=d.target)
    return images, text, train, test


def reproduce(out, steps):
    out.mkdir(parents=True, exist_ok=True)
    images, text, train, test = data()
    cfg = CONFIG
    held = test[:cfg["eval_count"]]
    np.savetxt(out / "held_out_indices.txt", held, fmt="%i")
    metadata = {"config": cfg | {"actual_steps": steps}, "torch": str(torch.__version__),
                "device": "cpu", "train_count": len(train), "test_count": len(test),
                "dataset": "UCI Optical Recognition of Handwritten Digits, sklearn load_digits"}
    records = []
    for name, (loops, shared) in VARIANTS.items():
        torch.manual_seed(cfg["seed"])
        model = build(loops, shared)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
        start = model.final.weight.detach().clone()
        batches = np.random.default_rng(cfg["seed"])
        torch.manual_seed(cfg["seed"] + 1)
        for step in range(steps):
            ix = batches.choice(train, cfg["batch"], replace=False)
            mask = torch.ones(len(ix), 1, dtype=torch.bool)
            opt.zero_grad(set_to_none=True)
            loss, _ = training_loss(model, images[ix], text[ix], mask)
            loss.backward()
            opt.step()
        change = (model.final.weight.detach() - start).norm().item()
        if not change > 0:
            raise RuntimeError("optimizer failed to update final projection")
        checkpoint = out / f"{name}.pt"
        torch.save(model.state_dict(), checkpoint)
        check = build(loops, shared)
        check.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
        check.eval()
        with torch.inference_mode():
            torch.manual_seed(cfg["seed"] + 2)
            noisy = .5 * images[held] + torch.randn_like(images[held])
            mask = torch.ones(len(held), 1, dtype=torch.bool)
            pred = check(noisy, text[held], mask)
            if not torch.equal(pred, model.eval()(noisy, text[held], mask)):
                raise RuntimeError("reloaded predictions differ")
            np.savez_compressed(out / f"{name}-predictions.npz", pred=pred.numpy(), ids=held)
            torch.manual_seed(cfg["seed"] + 3)
            digit_text = torch.eye(10).unsqueeze(1)
            samples = euler_sample(check, digit_text, torch.ones(10, 1, dtype=torch.bool),
                                   image_size=8, steps=cfg["sample_steps"], cfg_scale=1.)
            np.save(out / f"{name}-samples.npy", samples.numpy())
        records.append(dict(name=name, loops=loops, shared=shared, active_block_calls=2+loops,
                            parameters=sum(p.numel() for p in model.parameters()),
                            final_update_l2=change, last_train_loss=loss.item()))
    (out / "run.json").write_text(json.dumps(metadata | {"models":records}, indent=2))
    evaluate(out)


def evaluate(out):
    images, _, _, test = data()
    doc = json.loads((out / "run.json").read_text())
    summary = []
    sheet = Image.new("RGB", (452, len(doc["models"])*34), "white")
    draw = ImageDraw.Draw(sheet)
    for row, m in enumerate(doc["models"]):
        name = m["name"]
        datafile = np.load(out / f"{name}-predictions.npz")
        ids, pred = datafile["ids"], datafile["pred"]
        if not np.array_equal(ids, test[:CONFIG["eval_count"]]):
            raise ValueError("saved prediction IDs differ from expected holdout")
        mse = ((images[ids].numpy()-pred) ** 2).mean(axis=(1,2,3))
        summary.append(m | {"mean_mse_at_t_0_5":float(mse.mean()), "examples":len(ids)})
        samples = np.load(out / f"{name}-samples.npy")
        draw.text((2,row*34+9), name, fill="black")
        for digit, sample in enumerate(samples):
            tile = Image.fromarray(((sample[0]+1)*127.5).clip(0,255).astype("uint8"))
            sheet.paste(tile.resize((32,32),Image.Resampling.NEAREST),(92+digit*35,row*34))
    sheet.save(out / "comparison.png")
    with (out / "summary.csv").open("w",newline="") as h:
        w=csv.DictWriter(h,summary[0].keys())
        w.writeheader()
        w.writerows(summary)
    print((out / "summary.csv").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=["demo","reproduce","evaluate"])
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    torch.set_num_threads(CONFIG["threads"])
    out=args.out or ROOT / ("demo-output" if args.command=="demo" else "results")
    if args.command=="evaluate":
        evaluate(out)
    else:
        reproduce(out, 3 if args.command=="demo" else CONFIG["steps"])


if __name__ == "__main__":
    main()
