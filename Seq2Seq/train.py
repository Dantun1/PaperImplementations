import os

import torch
import torch.nn.functional as F

from Seq2Seq.data.dataset import EnglishFrenchDataset
from Seq2Seq.data.loader import create_loader
from Seq2Seq.data.vocab import Vocabulary
from Seq2Seq.utils.config import load_config
from Seq2Seq.utils.load_clean_df import load_and_clean_df
from models import DeepLSTMEncoder, DeepLSTMDecoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model Dimensions
config_path = os.path.join(BASE_DIR, "configs", "config.yaml")
cfg = load_config(config_path)

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

checkpoint_dir = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(checkpoint_dir, exist_ok=True)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0

    for batch_idx, (Xbe, Xbf, Yb) in enumerate(loader):
        # Get hidden vectors + cell states from encoder
        states = encoder(Xbe)
        # Pass through decoder to get logits
        logitsb, _ = decoder(Xbf, states)
        # Permute for torch API
        logitsb = logitsb.permute(0, 2, 1)
        loss = F.cross_entropy(logitsb, Yb, ignore_index=french_vocab.stoi["<PAD>"])

        # Optimisation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
        optimizer.step()

        epoch_loss += loss.item()

        if batch_idx % 100 == 0:
            print(f"Epoch {epoch + 1} | Batch {batch_idx}/{len(loader)} | Loss: {loss.item():.4f}")

    avg_loss = epoch_loss / len(loader)
    scheduler.step(avg_loss)

    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'vocab_src': eng_vocab,
        'vocab_tgt': french_vocab
    }, os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch + 1}.pt"))








