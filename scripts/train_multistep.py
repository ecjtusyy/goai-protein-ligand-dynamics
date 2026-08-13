import argparse
import csv
from pathlib import Path

import torch
from torch import nn

from src.geometry import bond_length_error, infer_bonds
from src.model import VelocityMLP
from src.training import load_split, seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--bond-weight", type=float, default=0.2)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", default="multistep_geometry")
    args = parser.parse_args()

    seed_everything(args.seed)
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/MISATO_100/raw"
    trajectories = load_split(raw / "MD.hdf5", raw / "train_MD.txt")

    checkpoint = torch.load(
        root / "checkpoints/one_step_mlp.pt",
        map_location="cpu",
        weights_only=True,
    )
    model = VelocityMLP(checkpoint["velocity_scale"], checkpoint["hidden_size"])
    model.load_state_dict(checkpoint["model_state"])
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scale = model.velocity_scale.detach()
    logs = []

    bonds = {
        pdb_id: torch.from_numpy(infer_bonds(positions[0].numpy(), atoms.numpy()))
        for pdb_id, positions, atoms in trajectories
    }

    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(trajectories))
        rollout_total = 0.0
        bond_total = 0.0
        model.train()

        for index in order.tolist():
            pdb_id, positions, atom_numbers = trajectories[index]
            max_start = len(positions) - args.horizon - 3
            start = int(torch.randint(max_start + 1, ()).item())
            history = positions[start : start + 3]
            target = positions[start + 3 : start + 3 + args.horizon]
            prediction = model.rollout(history, atom_numbers, args.horizon)

            rollout_loss = nn.functional.mse_loss(prediction / scale, target / scale)
            geometry_loss = bond_length_error(
                prediction,
                positions[0],
                bonds[pdb_id],
                squared=True,
            ) / scale.square()
            loss = rollout_loss + args.bond_weight * geometry_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            rollout_total += float(rollout_loss)
            bond_total += float(geometry_loss)

        mean_rollout = rollout_total / len(trajectories)
        mean_bond = bond_total / len(trajectories)
        logs.append(
            {
                "epoch": epoch,
                "rollout_loss": mean_rollout,
                "bond_loss": mean_bond,
                "total_loss": mean_rollout + args.bond_weight * mean_bond,
            }
        )
        print(
            f"epoch={epoch:02d}",
            f"rollout={mean_rollout:.6f}",
            f"bond={mean_bond:.6f}",
        )

    torch.save(
        {
            "model_state": model.state_dict(),
            "velocity_scale": float(model.velocity_scale),
            "hidden_size": checkpoint["hidden_size"],
            "horizon": args.horizon,
            "bond_weight": args.bond_weight,
            "seed": args.seed,
        },
        root / f"checkpoints/{args.name}.pt",
    )
    with (root / f"results/{args.name}_training.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=logs[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(logs)


if __name__ == "__main__":
    main()
