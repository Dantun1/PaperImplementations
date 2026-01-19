import torch
import torch.nn.functional as F

from Seq2Seq.data.dataset import EnglishFrenchDataset
from Seq2Seq.data.loader import create_loader
from Seq2Seq.data.vocab import Vocabulary
from Seq2Seq.utils.config import load_config
from Seq2Seq.utils.load_clean_df import load_and_clean_df
from models import DeepLSTMEncoder, DeepLSTMDecoder

# Model Dimensions

cfg = load_config("configs/config.yaml")

EMB_DIM = cfg["model"]["emb_dim"]
HIDDEN_DIM = cfg["model"]["hidden_dim"]
INIT_LR = cfg["training"]["init_lr"]
NUM_EPOCHS = cfg["training"]["num_epochs"]

# Initialise vocabularies

data = load_and_clean_df("../data/english_french.csv")

eng_vocab = Vocabulary.from_series(data["English"])
french_vocab = Vocabulary.from_series(data["French"])


ef_dataset = EnglishFrenchDataset(data,src_vocab=eng_vocab,target_vocab=french_vocab)

loader = create_loader(ef_dataset)

# Initialise Model

encoder = DeepLSTMEncoder(len(eng_vocab),EMB_DIM,HIDDEN_DIM)
decoder = DeepLSTMDecoder(len(french_vocab), EMB_DIM, HIDDEN_DIM)

# Initialise optimizer and LR scheduler

params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(
    params,
    lr=INIT_LR
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=500
)

# Training loop

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0

    for batch_idx, (Xbe, Xbf, Yb) in enumerate(loader):
        # Get hidden vectors from encoder
        states = encoder(Xbe)
        # Pass through decoder to get logits
        logitsb, _ = decoder(Xbf, states)
        # Permute for torch API
        logitsb = logitsb.permute(0, 2, 1)
        loss = F.cross_entropy(logitsb, Yb)

        # Optimisation
        optimizer.zero_grad()
        loss.backward()
        # Clip gradients
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()
        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    scheduler.step(avg_loss)








