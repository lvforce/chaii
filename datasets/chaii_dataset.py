import torch
from torch.utils.data import Dataset


class ChaiiDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': torch.tensor(self.data[idx]['input_id'], dtype=torch.long),
            'attention_mask': torch.tensor(self.data[idx]['attention_mask'], dtype=torch.long),
            'start_positions': torch.tensor(self.data[idx]['start_positions'], dtype=torch.long),
            'end_positions': torch.tensor(self.data[idx]['end_positions'], dtype=torch.long)
        }