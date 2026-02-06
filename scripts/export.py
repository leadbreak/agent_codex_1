import argparse

import torch

from hymba_plus.core.config import HymbaPlusConfig
from hymba_plus.models.language_model import HymbaPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hymba_plus.yaml")
    parser.add_argument("--out", default="hymba_plus.pt")
    args = parser.parse_args()

    config = HymbaPlusConfig.from_yaml(args.config)
    model = HymbaPlus(config)
    torch.save({"state_dict": model.state_dict(), "config": config}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
