import argparse

import torch

from hymba_plus.core.config import HymbaPlusConfig
from hymba_plus.models.language_model import HymbaPlus


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/hymba_plus.yaml")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args()

    config = HymbaPlusConfig.from_yaml(args.config)
    model = HymbaPlus(config)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.training.rl.learning_rate)

    for step in range(args.steps):
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        out = model(input_ids)
        reward = out.logits.mean()
        loss = -reward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        print(f"rl step={step} loss={loss.item():.4f}")


if __name__ == "__main__":
    main()
