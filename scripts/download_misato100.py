from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    snapshot_download(
        repo_id="chao1224/NeuralMD",
        repo_type="dataset",
        allow_patterns=["MISATO_100/raw/*"],
        local_dir=root / "data",
    )
    print(root / "data/MISATO_100/raw/MD.hdf5")


if __name__ == "__main__":
    main()
