import argparse

import torch

from hymba_plus.core.config import HymbaPlusConfig
from hymba_plus.models.language_model import HymbaPlus
from training.sft.trainer import SFTConfig, SFTTrainer


def make_dataloader(config: HymbaPlusConfig, steps: int):
    for _ in range(steps):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        yield {"input_ids": input_ids, "labels": labels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hymba_plus.yaml")
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args()

    config = HymbaPlusConfig.from_yaml(args.config)
    model = HymbaPlus(config)

    trainer = SFTTrainer(
        model,
        SFTConfig(
            max_steps=args.steps,
            warmup_steps=min(10, args.steps),
            learning_rate=config.training.sft.learning_rate,
            grad_accum_steps=1,
            bf16=True,
        ),
    )
    trainer.train(make_dataloader(config, args.steps))


if __name__ == "__main__":
    main()
