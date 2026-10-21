"""
generate_params_h.py
=====================
Genere DEUX fichiers a partir de hls_export/ :

  config.h  -> toutes les constantes de dimension (EMBED_DIM, NUM_HEADS...)
  params.h  -> UN SEUL tableau plat WEIGHTS[], et une constante "OFFSET_xxx"
               nommee pour chaque tenseur (plus AUCUN calcul d'offset a la main
               dans vitis_top.cpp)

IMPORTANT - deux conventions de poids cohabitent dans le projet, et ce script
les respecte chacune :

  1) patch_embed, pos_embed, depth_head : ces fonctions font `out = in @ W^T`
     (comme un nn.Linear PyTorch) -> on exporte les poids TELS QUELS, sans
     transposition.

  2) transformer_block (via la fonction matmul(A,B,C) = A @ B) : cette
     fonction attend les poids en (dim_entree, dim_sortie), soit l'INVERSE
     du stockage PyTorch (dim_sortie, dim_entree) -> on doit TRANSPOSER
     Wq, Wk, Wv (en fusionnant aussi les 2 tetes), Wo, W1 et W2.


"""

import json
from pathlib import Path
import numpy as np

OUT_DIR = Path("hls_export")
WEIGHTS_DIR = OUT_DIR / "weights"
CONFIG = json.load(open(OUT_DIR / "config.json"))

EMBED_DIM = CONFIG["embed_dim"]
NUM_HEADS = CONFIG["num_heads"]
HEAD_DIM = CONFIG["head_dim"]
NUM_BLOCKS = CONFIG["num_blocks"]
FF_DIM = CONFIG["ff_hidden_dim"]
PATCH_SIZE = CONFIG["patch_size"]
IN_CHANNELS = CONFIG["in_channels"]
HP = CONFIG["Hp"]
WP = CONFIG["Wp"]
NUM_PATCHES = CONFIG["num_patches"]
IMAGE_SIZE = CONFIG["image_size"]


def read_txt(path):
    with open(path) as f:
        shape = [int(x) for x in f.readline().split()]
        vals = [float(x) for x in f.readline().split()]
    return np.array(vals, dtype=np.float32).reshape(shape)


weights_list = []
offsets = {}


def add(name, arr):
    flat = np.asarray(arr, dtype=np.float32).flatten()
    offsets[name] = len(weights_list)
    weights_list.extend(flat.tolist())


def fused_qkv(block_idx, proj_name):
    """Concatene TOUTES les tetes et transpose chacune : (head_dim,embed_dim)
    -> (embed_dim,head_dim), puis hstack -> (embed_dim,embed_dim), pret pour
    matmul(A,B)=A@B."""
    heads_t = []
    for h in range(NUM_HEADS):
        w = read_txt(WEIGHTS_DIR / f"blocks.{block_idx}.attention.heads.{h}.{proj_name}.weight.txt")
        heads_t.append(w.T)
    return np.hstack(heads_t)


# ------------------------------------------------------------------
# 1. Patch embedding (convention native PyTorch, pas de transpose)
# ------------------------------------------------------------------
add("PATCH_EMBED_W", read_txt(WEIGHTS_DIR / "patch_embed.proj.weight.txt"))
add("PATCH_EMBED_B", read_txt(WEIGHTS_DIR / "patch_embed.proj.bias.txt"))

# ------------------------------------------------------------------
# 2. Positional embedding (lookup direct, pas de transpose)
# ------------------------------------------------------------------
add("POS_ROW_TABLE", read_txt(WEIGHTS_DIR / "pos_embed.row_embed.weight.txt"))
add("POS_COL_TABLE", read_txt(WEIGHTS_DIR / "pos_embed.col_embed.weight.txt"))

# ------------------------------------------------------------------
# 3. Blocs transformer (convention matmul, transpose + fusion des tetes)
# ------------------------------------------------------------------
for b in range(NUM_BLOCKS):
    add(f"BLOCK{b}_WQ", fused_qkv(b, "W_q"))
    add(f"BLOCK{b}_WK", fused_qkv(b, "W_k"))
    add(f"BLOCK{b}_WV", fused_qkv(b, "W_v"))
    add(f"BLOCK{b}_WO", read_txt(WEIGHTS_DIR / f"blocks.{b}.attention.proj.weight.txt").T)
    add(f"BLOCK{b}_BO", read_txt(WEIGHTS_DIR / f"blocks.{b}.attention.proj.bias.txt"))
    add(f"BLOCK{b}_W1", read_txt(WEIGHTS_DIR / f"blocks.{b}.ff.feature.0.weight.txt").T)
    add(f"BLOCK{b}_B1", read_txt(WEIGHTS_DIR / f"blocks.{b}.ff.feature.0.bias.txt"))
    add(f"BLOCK{b}_W2", read_txt(WEIGHTS_DIR / f"blocks.{b}.ff.feature.2.weight.txt").T)
    add(f"BLOCK{b}_B2", read_txt(WEIGHTS_DIR / f"blocks.{b}.ff.feature.2.bias.txt"))
    add(f"BLOCK{b}_NORM1_G", read_txt(WEIGHTS_DIR / f"blocks.{b}.norm1.weight.txt"))
    add(f"BLOCK{b}_NORM1_B", read_txt(WEIGHTS_DIR / f"blocks.{b}.norm1.bias.txt"))
    add(f"BLOCK{b}_NORM2_G", read_txt(WEIGHTS_DIR / f"blocks.{b}.norm2.weight.txt"))
    add(f"BLOCK{b}_NORM2_B", read_txt(WEIGHTS_DIR / f"blocks.{b}.norm2.bias.txt"))

# ------------------------------------------------------------------
# 4. Norm finale + depth head (convention native, pas de transpose)
# ------------------------------------------------------------------
add("FINAL_NORM_G", read_txt(WEIGHTS_DIR / "norm.weight.txt"))
add("FINAL_NORM_B", read_txt(WEIGHTS_DIR / "norm.bias.txt"))
add("DEPTH_HEAD_W", read_txt(WEIGHTS_DIR / "depth_head.proj.weight.txt"))
add("DEPTH_HEAD_B", read_txt(WEIGHTS_DIR / "depth_head.proj.bias.txt"))

# ------------------------------------------------------------------
# Ecriture de params.h
# ------------------------------------------------------------------
with open("params.h", "w") as f:
    f.write("#ifndef PARAMS_H\n#define PARAMS_H\n\n")
    f.write('#include "codes/types.h"\n\n')   # [AJOUT] pour le type data_t (float ou ap_fixed)
    f.write(f"const int TOTAL_WEIGHTS = {len(weights_list)};\n\n")
    for name, off in offsets.items():
        f.write(f"#define OFFSET_{name} {off}\n")

    # [MULTI-ARCH 2026-07-15] Tables d'offsets indexees PAR BLOC, pour que
    # vitis_top puisse choisir les poids du bloc b (boucle iterative for(b),
    # ou etage b du pipeline). Les #define OFFSET_BLOCK{b}_* ci-dessus sont
    # CONSERVES (rien supprime) ; ceci ne fait qu'en dresser la liste ordonnee.
    f.write("\n// Tables d'offsets par bloc (archis iteratif / pipeline)\n")
    _tensors = ["WQ", "WK", "WV", "WO", "BO", "W1", "B1", "W2", "B2",
                "NORM1_G", "NORM1_B", "NORM2_G", "NORM2_B"]
    for _t in _tensors:
        _vals = ", ".join(str(offsets[f"BLOCK{b}_{_t}"]) for b in range(NUM_BLOCKS))
        f.write(f"const int OFF_{_t}[{NUM_BLOCKS}] = {{ {_vals} }};\n")
    f.write("\n")

    # f.write("\nconst float WEIGHTS[TOTAL_WEIGHTS] = {\n")
    f.write("\nconst data_t WEIGHTS[TOTAL_WEIGHTS] = {\n")   # [AJOUT] data_t au lieu de float
    for i in range(0, len(weights_list), 10):
        ligne = weights_list[i:i + 10]
        f.write("    " + ", ".join(f"{v:.8f}f" for v in ligne) + ",\n")
    # f.write("};\n\n#endif\n")
    f.write("};\n\n// Genere par generate_params_h.py (type data_t defini dans codes/types.h)\n#endif\n")   # [AJOUT]

# ------------------------------------------------------------------
# Ecriture de config.h
# [MULTI-ARCH 2026-07-15] On ecrit config.h aux DEUX emplacements :
#   - "config.h"        (racine, historique)
#   - "codes/config.h"  (celui que le compilateur lit vraiment : vitis_top.cpp
#                         fait #include "config.h" depuis codes/).
# Avant, seul le fichier racine etait genere -> codes/config.h restait a la main
# et derivait (ex : NUM_BLOCKS 2 vs 3) -> #error a la csim. Ecrire les deux
# garantit qu'ils restent synchronises.
# ------------------------------------------------------------------
_config_txt = "#ifndef CONFIG_H\n#define CONFIG_H\n\n"
_config_txt += f"#define IMAGE_SIZE {IMAGE_SIZE}\n"
_config_txt += f"#define PATCH_SIZE {PATCH_SIZE}\n"
_config_txt += f"#define IN_CHANNELS {IN_CHANNELS}\n"
_config_txt += f"#define EMBED_DIM {EMBED_DIM}\n"
_config_txt += f"#define NUM_HEADS {NUM_HEADS}\n"
_config_txt += f"#define HEAD_DIM {HEAD_DIM}\n"
_config_txt += f"#define NUM_BLOCKS {NUM_BLOCKS}\n"
_config_txt += f"#define FF_DIM {FF_DIM}\n"
_config_txt += f"#define HP {HP}\n"
_config_txt += f"#define WP {WP}\n"
_config_txt += f"#define NUM_PATCHES {NUM_PATCHES}\n"
_config_txt += "\n#endif\n"
for _path in ("config.h", "codes/config.h"):
    with open(_path, "w") as f:
        f.write(_config_txt)

print(f"params.h genere : {len(weights_list)} valeurs, {len(offsets)} tenseurs nommes")
print("config.h genere (racine + codes/).")

