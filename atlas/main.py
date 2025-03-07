import hydra
import torch
from datasets import load_dataset
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaConfig, LlamaForCausalLM
from utils import log_config


@hydra.main(
    config_path="config",
    config_name="default.yaml",
    version_base="1.3",
)
def main(cfg: DictConfig):
    OmegaConf.set_struct(cfg, False)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    log_config(cfg_dict)

    config = LlamaConfig(
        vocab_size=32000,
        hidden_size=128,
        intermediate_size=512,
        num_attention_heads=4,
        num_hidden_layers=4,
        max_position_embeddings=256,
    )

    model = LlamaForCausalLM(config)
    print(model.num_parameters())

    dataset = load_dataset("tiny_shakespeare")["train"]["text"][0]
    chars = sorted(list(set(dataset)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}

    class TSDataset(Dataset):
        def __init__(self, text, seq_len=32):
            self.data = torch.tensor([char_to_idx[c] for c in text], dtype=torch.long)
            self.seq_len = seq_len

        def __len__(self):
            return len(self.data) - self.seq_len

        def __getitem__(self, idx):
            input_ids = self.data[idx : idx + self.seq_len]
            labels = self.data[idx + 1 : idx + self.seq_len + 1]
            return input_ids, labels

    toy_dataset = TSDataset(dataset)
    dataloader = DataLoader(toy_dataset, batch_size=16, shuffle=True)

    for input_ids, labels in dataloader:
        output = model(input_ids, labels=labels)
        print(f"Loss: {output['loss'].item()}")
        break
    breakpoint()


if __name__ == "__main__":
    main()
