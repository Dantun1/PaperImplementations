# Change the range to generate more / less, change index of eng sentence for other starting point
import torch

for i in range(50):
    eng_sentence = torch.flip(Xte_en[1500 + i],dims = [0])
    mask = (eng_sentence != 0)
    eng_sentence = eng_sentence[mask].view(1,-1)
    states = encoder(eng_sentence)

    curr_token = torch.tensor([[1]])

    decoded_indices = []

    max_generation_len = 50

    for i in range(max_generation_len):
        logits, states = decoder(curr_token, states)
        probs = F.softmax(logits.squeeze() / 0.1, dim=0)

        next_token_idx = torch.argmax(probs).item()

        if next_token_idx == 2:
            break

        decoded_indices.append(next_token_idx)

        curr_token = torch.tensor([[next_token_idx]])

    print(''.join(reversed(list(decode_text(eng_sentence.tolist()[0],english = True)))))
    print(decode_text(decoded_indices), end = "\n\n")
