import torch


class DeepLSTMDecoder(torch.nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_dim:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.cells = torch.nn.ModuleList([
            torch.nn.LSTMCell(embedding_dim, hidden_dim),
            torch.nn.LSTMCell(hidden_dim, hidden_dim),
        ])
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, encoder_states):
        Xen = self.embedding(x)
        B, T, C = Xen.shape

        # Initialise states of each layer to corresponding encoder layer states
        states = [s for s in encoder_states]
        outputs = []

        # Through time
        for t in range(T):
            inputx = Xen[:, t, :]
            # Through layers
            for i in range(len(self.cells)):
                h_prev, c_prev = states[i]
                h, c = self.cells[i](inputx, (h_prev,c_prev))
                states[i] = (h,c)
                inputx = h
            # inputx becomes the final hidden state, so pass through the output linear layer to get logits
            logits_example = self.out(inputx)
            outputs.append(logits_example)

        return torch.stack(outputs, dim=1), states
