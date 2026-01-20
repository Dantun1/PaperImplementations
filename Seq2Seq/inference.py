import torch
import torch.nn.functional as F
import os
import sys
import pandas as pd
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Seq2Seq.utils.config import load_config
from Seq2Seq.utils.load_clean_df import load_and_clean_df
from Seq2Seq.data.vocab import Vocabulary
from models import DeepLSTMEncoder, DeepLSTMDecoder

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(BASE_DIR, "configs", "config.yaml")
CHECKPOINT_PATH = os.path.join(BASE_DIR, "checkpoints", "checkpoint_epoch_10.pt")
DATA_PATH = os.path.join(BASE_DIR, "..", "data", "english_french.csv")

cfg = load_config(CONFIG_PATH)


data = load_and_clean_df(DATA_PATH)
eng_vocab = Vocabulary.from_series(data["English"])
french_vocab = Vocabulary.from_series(data["French"])

encoder = DeepLSTMEncoder(len(eng_vocab), cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"])
decoder = DeepLSTMDecoder(len(french_vocab), cfg["model"]["emb_dim"], cfg["model"]["hidden_dim"])

checkpoint = torch.load(CHECKPOINT_PATH, weights_only=False)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
decoder.load_state_dict(checkpoint['decoder_state_dict'])


def translate_sentence(sentence: str, temperature=1, max_len=80):
    """
    Translates a raw English string to French.
    """
    indices = eng_vocab.get_indices(sentence)

    tensor_en = torch.LongTensor(indices).unsqueeze(0)
    tensor_en = torch.flip(tensor_en, dims=[1])

    with torch.no_grad():
        states = encoder(tensor_en)

    start_token = french_vocab.stoi["<SOS>"]
    decoder_input = torch.LongTensor([[start_token]])

    decoded_indices = []

    for _ in range(max_len):
        with torch.no_grad():
            logits, states = decoder(decoder_input, states)

            logits = logits.squeeze(0) / temperature
            probs = F.softmax(logits, dim=-1)

            next_token = torch.argmax(probs, dim=-1).item()

            if next_token == french_vocab.stoi["<EOS>"]:
                break

            decoded_indices.append(next_token)

            decoder_input = torch.LongTensor([[next_token]])

    translated_words = french_vocab.get_string(decoded_indices)
    return "".join(translated_words)


if __name__ == "__main__":
    print("Type a sentence to translate")
    print("Type 'random' to see a random example")
    print("Type 'q' to quit\n")

    while True:
        user_input = input("English Input:")

        if user_input.lower() == 'q':
            break

        elif user_input.lower() == 'random':
            idx = random.randint(0, len(data) - 1)
            en_text = data.iloc[idx]["English"]
            fr_true = data.iloc[idx]["French"]

            pred = translate_sentence(en_text)

            print(f"Original:  {en_text}")
            print(f"Target:    {fr_true}")
            print(f"Predicted: {pred}")
        else:
            pred = translate_sentence(user_input)
            print(f"French: {pred}")
            print("-" * 30)