import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from transformer import DepthTransformer
import matplotlib.pyplot as plt

from dataset import NYUDepthDataset


def entrainer(model, dataloader, optimizer, criterion, device, nb_epochs, checkpoint_dir=None):
    model.train()
    historique_loss = []

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(nb_epochs):
        total_loss = 0.0
        for images, depths in dataloader:
            images = images.to(device, non_blocking=True)
            depths = depths.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(images)          # (B, H, W)
            loss = criterion(predictions, depths)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        moyenne = total_loss / len(dataloader)
        historique_loss.append(moyenne)
        print(f"Epoch {epoch+1}/{nb_epochs} - loss: {moyenne:.4f}")

        # --- sauvegarde du modèle après chaque epoch ---
        if checkpoint_dir is not None:
            checkpoint = {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": moyenne,
            }
            # checkpoint numéroté (historique complet)
            chemin_epoch = checkpoint_dir / f"checkpoint_epoch_{epoch+1:03d}.pth"
            torch.save(checkpoint, chemin_epoch)
            # toujours écraser un "last.pth" pratique à recharger pour test.py
            torch.save(checkpoint, checkpoint_dir / "last.pth")
            print(f"  -> checkpoint sauvegardé : {chemin_epoch}")

    return historique_loss


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device utilisé : {device}")

    # --- hyperparamètres ---
    IMAGE_SIZE = 64       # doit être divisible par PATCH_SIZE
    PATCH_SIZE = 16
    EMBED_DIM = 32
    NUM_HEADS = 2
    NUM_BLOCKS = 3
    BATCH_SIZE = 8
    NB_EPOCHS = 10
    LR = 3e-4
    

    # --- dataset NYU Depth V2 (fichiers locaux) ---
    DATA_ROOT = "D:\Devoir\Stage1\Tuto\FichierTutoTransformer\python"
    CHECKPOINT_DIR = "D:\Devoir\Stage1\Tuto\FichierTutoTransformer\python\checkpoints"

    print("Indexation des paires rgb/depth (peut prendre quelques secondes)...")
    dataset_train = NYUDepthDataset(f"{DATA_ROOT}/train", image_size=IMAGE_SIZE)
    dataset_val = NYUDepthDataset(f"{DATA_ROOT}/test/official", image_size=IMAGE_SIZE)

    dataloader = DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=4,
    )
    dataloader_val = DataLoader(
        dataset_val, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True,
    )

    print(f"Train: {len(dataset_train)} paires - Val: {len(dataset_val)} paires")

    # --- modèle ---
    model = DepthTransformer(
        embed_dim=EMBED_DIM,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        patch_size=PATCH_SIZE,
        in_channels=3,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()   # à terme : SiLog loss, souvent meilleure pour la profondeur

    # --- sauvegarde de la config du modèle pour pouvoir le recharger dans test.py ---
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)
    config = {
        "image_size": IMAGE_SIZE,
        "patch_size": PATCH_SIZE,
        "embed_dim": EMBED_DIM,
        "num_heads": NUM_HEADS,
        "num_blocks": NUM_BLOCKS,
        "in_channels": 3,
        "data_root": DATA_ROOT,
    }
    with open(Path(CHECKPOINT_DIR) / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    historique = entrainer(
        model, dataloader, optimizer, criterion, device, NB_EPOCHS,
        checkpoint_dir=CHECKPOINT_DIR,
    )

    plt.plot(historique)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title("Entraînement - estimation de profondeur")
    plt.savefig("loss_profondeur.png")
    plt.show()