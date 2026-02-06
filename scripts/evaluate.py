import argparse

import torch

from hymba_plus.core.config import HymbaPlusConfig
from hymba_plus.models.language_model import HymbaPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hymba_plus.yaml")
    args = parser.parse_args()

    config = HymbaPlusConfig.from_yaml(args.config)
    model = HymbaPlus(config)
    model.eval()

    input_ids = torch.randint(0, config.vocab_size, (2, 16))
    out = model(input_ids)
    print("로그잇 텐서 크기:", out.logits.shape)


if __name__ == "__main__":
    main()
