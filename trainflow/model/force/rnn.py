import torch.nn as nn


class RNN(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_dim,
        layer_num,
        n_emb,
    ):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, layer_num, batch_first=True)
        self.fc = nn.Linear(hidden_dim, n_emb)

    def get_optim_groups(self, weight_decay: float = 1e-3):
        return [{"params": self.parameters(), "weight_decay": weight_decay}]

    def forward(self, sample):
        x, _ = self.rnn(sample)
        x = self.fc(x)
        return x
