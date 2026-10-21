import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from pathlib import Path
from PIL import Image
import numpy as np
import json


# ============================================================
#  Dataset NYU Depth V2 (fichiers locaux, structure Kaggle
#  "awsaf49/nyuv2-official-split-dataset") :
#
#    train/<scene>/rgb_NNNNN.png + depth_NNNNN.png
#    test/official/rgb_NNNNN.png + depth_NNNNN.png
#
#  Profondeur : PNG 16-bit (mode 'I' en PIL), valeur en
#  millimètres -> on divise par 1000 pour obtenir des mètres.
#
#  L'indexation (parcours disque + appariement rgb/depth) est
#  coûteuse sur un gros dataset -> mise en cache dans un .json
#  à côté du dossier scanné, réutilisé aux lancements suivants.
# ============================================================
class NYUDepthDataset(Dataset):
    def __init__(self, root_dir, image_size, cache_file=None):
        self.root_dir = Path(root_dir)
        self.image_size = image_size

        if cache_file is None:
            cache_file = self.root_dir.parent / f"{self.root_dir.name}_index.json"
        cache_file = Path(cache_file)

        if cache_file.exists():
            print(f"Index trouvé en cache : {cache_file}")
            with open(cache_file) as f:
                pairs_str = json.load(f)
            self.pairs = [(Path(r), Path(d)) for r, d in pairs_str]
        else:
            print(f"Pas de cache, indexation de {root_dir}...")
            self.pairs = []
            for rgb_path in sorted(self.root_dir.rglob("rgb_*.png")):
                if ".ipynb_checkpoints" in rgb_path.parts:
                    continue
                depth_path = rgb_path.with_name(rgb_path.name.replace("rgb_", "depth_"))
                if depth_path.exists():
                    self.pairs.append((rgb_path, depth_path))

            with open(cache_file, "w") as f:
                json.dump([[str(r), str(d)] for r, d in self.pairs], f)
            print(f"Index sauvegardé dans {cache_file} ({len(self.pairs)} paires)")

        if len(self.pairs) == 0:
            raise RuntimeError(f"Aucune paire rgb/depth trouvée dans {root_dir}")

        self.image_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),   # -> (3, H, W), valeurs dans [0, 1]
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        rgb_path, depth_path = self.pairs[idx]

        image = Image.open(rgb_path).convert("RGB")
        image = self.image_transform(image)                       # (3, H, W)

        depth_img = Image.open(depth_path)                        # mode 'I', 16-bit
        depth_arr = np.array(depth_img, dtype=np.float32) / 1000.0  # -> mètres
        depth = torch.from_numpy(depth_arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        depth = F.interpolate(depth, size=(self.image_size, self.image_size),
                               mode="bilinear", align_corners=False)
        depth = depth.squeeze(0).squeeze(0)                        # (H, W)

        return image, depth