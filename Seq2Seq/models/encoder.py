import torch


class DeepLSTMEncoder(torch.nn.Module):
    def __init__(self, vocab_size:int, embedding_dim:int, hidden_dim:int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self.cells = torch.nn.ModuleList([
            torch.nn.LSTMCell(embedding_dim, hidden_dim),
            torch.nn.LSTMCell(hidden_dim, hidden_dim),
        ])

    def forward(self, x):
        Xen = self.embedding(x)
        B, T, C = Xen.shape

        # Initialise states list to (h,c) zero tensors per layer
        states = []
        for _ in range(len(self.cells)):
            h, c = torch.zeros(B, self.hidden_dim), torch.zeros(B, self.hidden_dim)
            states.append((h,c))

        # Through time
        for t in range(T):
            inputx = Xen[:, t, :]
            # Through layers, input to the upper layer is hidden of lower
            for i in range(len(self.cells)):
                # Update states after passing through cell at layer i
                h_prev, c_prev = states[i]
                h, c = self.cells[i](inputx, (h_prev,c_prev))
                states[i] = (h,c)
                inputx = h

        return states
