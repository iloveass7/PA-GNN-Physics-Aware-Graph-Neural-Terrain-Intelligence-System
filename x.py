import torch
ckpt = torch.load("checkpoints/mae_best.pt", map_location="cpu", weights_only=False)
print(ckpt.keys())                          # top-level keys
enc = ckpt.get("encoder_state_dict", ckpt)
print(list(enc.keys())[:10])               # first 10 weight keys