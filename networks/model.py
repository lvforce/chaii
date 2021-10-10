import torch.nn as nn
from transformers import  AutoModel, AutoConfig

class Model(nn.Module):
    def __init__(self, model_name):
        super(Model, self).__init__()
        self.model_name = model_name
        self.config = AutoConfig.from_pretrained(self.model_name)
        self.roberta = AutoModel.from_pretrained(self.model_name, config=self.config)
        self.roberta.pooler = nn.Identity()
        self.linear = nn.Linear(self.config.hidden_size, 2)

    def loss_fn(self, start_logits, end_logits, start_positions, end_positions):
        if len(start_positions.size()) > 1:
            start_positions = start_positions.squeeze(-1)
        if len(end_positions.size()) > 1:
            end_positions = end_positions.squeeze(-1)

        ignored_index = start_logits.size(1)
        start_positions = start_positions.clamp(0, ignored_index)
        end_positions = end_positions.clamp(0, ignored_index)
        loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
        start_loss = loss_fct(start_logits, start_positions)
        end_loss = loss_fct(end_logits, end_positions)
        total_loss = 0.75 * start_loss + 0.25 * end_loss

        return total_loss

    def forward(self, **xb):
        x = self.roberta(input_ids=xb['input_ids'], attention_mask=xb['attention_mask'])[0]
        x = self.linear(x)

        start_logits, end_logits = x.split(1, dim=1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()
        start_positions = xb['start_positions']
        end_positions = xb['end_positions']

        loss = None

        if start_positions is not None and end_positions is not None:
            loss = self.loss_fn(start_logits, end_logits, start_positions, end_positions)

        return (start_logits, end_logits), loss