import torch

lr=1e-3
params = list(encoder.parameters()) + list(decoder.parameters())
optimizer = torch.optim.Adam(
    params,
    lr=lr
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.1, patience=500
)

MAX_STEPS = 20000
losses = []
updates = []

for step in range(MAX_STEPS):
    # Get batch tensors
    ix_b = torch.randint(0,Xtr_en.shape[0], size = (64,))
    Xbe,Xbf,Yb = Xtr_en[ix_b], Xtr_fr[ix_b], Ytr[ix_b]

    Xbe_reversed = torch.flip(Xbe, dims=[1])
    # Get hidden vectors from encoder
    states = encoder(Xbe_reversed)
    # Pass through decoder to get logits
    logitsb, _ = decoder(Xbf,states)
    # Permute for torch API
    logitsb= logitsb.permute(0,2,1)
    loss = F.cross_entropy(logitsb, Yb)

    # Optimisation
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients
    torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
    optimizer.step()
    scheduler.step(loss.item())
    # Track stats
    current_lr = optimizer.param_groups[0]['lr']

    losses.append(loss.item())
    with torch.no_grad():
        updates.append([((current_lr*p.grad).std()/p.data.std()).log10() for p in params])
    if step % 200 == 0:
        print(f"Step {step}: {loss.item()}")
