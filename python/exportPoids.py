import json
from pathlib import Path

import torch

from transformer import DepthTransformer, attention

# ------------------------------------------------------------------
# 1. Configuration du mini-modele ("Petit")
# ------------------------------------------------------------------
IMAGE_SIZE = 64
PATCH_SIZE = 16
EMBED_DIM = 32
NUM_HEADS = 2
# NUM_BLOCKS = 2   # [MODIF Claude 2026-07-15] -> 3 (le modele entraine a 3 blocs)
NUM_BLOCKS = 3
IN_CHANNELS = 3

# [MODIF Claude 2026-07-15] Checkpoint du modele ENTRAINE (poids reels, plus d'aleatoire).
CHECKPOINT = Path("checkpoints/last.pth")

OUT_DIR = Path("hls_export")
WEIGHTS_DIR = OUT_DIR / "weights"
REF_DIR = OUT_DIR / "reference"


def dump_tensor(path: Path, tensor: torch.Tensor):
    """Ecrit un tenseur dans un fichier texte : shape puis valeurs aplaties."""
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = tensor.detach().cpu().reshape(-1)
    shape = list(tensor.shape)
    with open(path, "w") as f:
        f.write(" ".join(str(d) for d in shape) + "\n")
        f.write(" ".join(f"{v:.8f}" for v in arr.tolist()) + "\n")


def main():
    torch.manual_seed(42)

    model = DepthTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        in_channels=IN_CHANNELS,
    )

    ckpt = torch.load(CHECKPOINT, map_location="cpu")
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    if isinstance(ckpt, dict) and "epoch" in ckpt:
        print(f"Poids entraines charges depuis {CHECKPOINT} (epoch {ckpt['epoch']}, loss {ckpt['loss']:.4f})")
    else:
        print(f"Poids entraines charges depuis {CHECKPOINT}")

    model.eval()  # important : desactive le dropout (pas d'equivalent en HLS)

    Hp = IMAGE_SIZE // PATCH_SIZE
    Wp = IMAGE_SIZE // PATCH_SIZE

    # --- config.json : toutes les dimensions dont le C++ aura besoin ---
    config = {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "head_dim": EMBED_DIM // NUM_HEADS,
        "num_blocks": NUM_BLOCKS,
        "in_channels": IN_CHANNELS,
        "Hp": Hp,
        "Wp": Wp,
        "num_patches": Hp * Wp,
        "ff_hidden_dim": EMBED_DIM * 4,
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config sauvegardee : {OUT_DIR / 'config.json'}")

    # ----------------------------------------------------------------
    # 2. Export de tous les poids (nn.Parameter) du modele
    # ----------------------------------------------------------------
    n_params = 0
    for name, param in model.named_parameters():
        dump_tensor(WEIGHTS_DIR / f"{name}.txt", param)
        n_params += param.numel()
    print(f"Poids exportes dans {WEIGHTS_DIR}/ ({n_params} valeurs au total)")

    # ----------------------------------------------------------------
    # 3. Forward manuel sur UNE image de test fixe, en dumpant CHAQUE
    #    resultat intermediaire -> reference/ pour les testbenchs C++
    # ----------------------------------------------------------------
    torch.manual_seed(123)
    image = torch.rand(1, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
    dump_tensor(REF_DIR / "input_image.txt", image)

    with torch.no_grad():
        # --- patch embedding ---
        x, Hp, Wp = model.patch_embed(image)
        dump_tensor(REF_DIR / "patch_embed_output.txt", x)

        # --- position embedding ---
        pos = model.pos_embed(Hp, Wp, image.device)
        dump_tensor(REF_DIR / "pos_embed_output.txt", pos)

        x = x + pos.unsqueeze(0)
        dump_tensor(REF_DIR / "block0_input.txt", x)

        # --- blocs transformer ---
        for i, block in enumerate(model.blocks):
            head_outputs = []
            for h, head in enumerate(block.attention.heads):
                Q = head.W_q(x)
                K = head.W_k(x)
                V = head.W_v(x)
                dump_tensor(REF_DIR / f"block{i}_head{h}_Q.txt", Q)
                dump_tensor(REF_DIR / f"block{i}_head{h}_K.txt", K)
                dump_tensor(REF_DIR / f"block{i}_head{h}_V.txt", V)

                out_h = attention(Q, K, V, head.W_q.out_features, causal=head.causal)
                dump_tensor(REF_DIR / f"block{i}_head{h}_output.txt", out_h)
                head_outputs.append(out_h)

            concat = torch.cat(head_outputs, dim=-1)
            attn_out = block.attention.proj(concat)
            dump_tensor(REF_DIR / f"block{i}_attention_output.txt", attn_out)

            x = block.norm1(x + block.dropout(attn_out))
            dump_tensor(REF_DIR / f"block{i}_after_norm1.txt", x)

            ff_out = block.ff(x)
            dump_tensor(REF_DIR / f"block{i}_ff_output.txt", ff_out)

            x = block.norm2(x + block.dropout(ff_out))
            dump_tensor(REF_DIR / f"block{i}_after_norm2.txt", x)

            if i + 1 < NUM_BLOCKS:
                dump_tensor(REF_DIR / f"block{i + 1}_input.txt", x)

        # --- norm finale + tete de profondeur ---
        x = model.norm(x)
        dump_tensor(REF_DIR / "final_norm_output.txt", x)

        depth = model.depth_head(x, Hp, Wp)
        dump_tensor(REF_DIR / "final_depth_output.txt", depth)

    print(f"Valeurs de reference exportees dans {REF_DIR}/")
    print("Termine.")


if __name__ == "__main__":
    main()
