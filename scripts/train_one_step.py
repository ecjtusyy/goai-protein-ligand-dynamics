import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from src.model import VelocityMLP
from src.training import load_split, seed_everything, velocity_examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/MISATO_100/raw"
    trajectories = load_split(raw / "MD.hdf5", raw / "train_MD.txt")
    previous, current, target, atoms = velocity_examples(trajectories)
    velocity_scale = float(target.std().clamp_min(1e-6))

    model = VelocityMLP(velocity_scale)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    losses = []

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(target))
        total = 0.0
        model.train()

        for batch in order.split(args.batch_size):
            prediction = model.predict_velocity(
                previous[batch],
                current[batch],
                atoms[batch],
            )
            loss = nn.functional.mse_loss(prediction / velocity_scale, target[batch] / velocity_scale)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)

        mean_loss = total / len(target)
        losses.append({"epoch": epoch, "loss": mean_loss})
        print(f"epoch={epoch:02d} loss={mean_loss:.6f}")

    checkpoints = root / "checkpoints"
    results = root / "results"
    checkpoints.mkdir(exist_ok=True)
    results.mkdir(exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "velocity_scale": velocity_scale,
            "hidden_size": 64,
            "seed": args.seed,
        },
        checkpoints / "one_step_mlp.pt",
    )
    with (results / "one_step_training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "loss"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(losses)


if __name__ == "__main__":
    main()
