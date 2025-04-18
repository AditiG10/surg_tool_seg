import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

# Suppress OpenMP duplicate lib warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Add parent directory to path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SurgicalDataset, get_training_augmentations
from src.models.deeplabv3 import get_deeplabv3

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)


def run_training(
    data_root='data/train',
    n_classes=10,
    epochs=5,
    batch_size=2,
    lr=1e-4,
    cpu_mode=False
):
    device = torch.device('cpu' if cpu_mode else ('cuda' if torch.cuda.is_available() else 'cpu'))
    img_sz = (256, 256)

    # Use only first 5 videos for quick training
    all_videos = sorted([
        d for d in os.listdir(data_root)
        if d.startswith("video_") and os.path.isdir(os.path.join(data_root, d))
    ])[:5]

    print(f"\nUsing {len(all_videos)} training videos from: {data_root}\n")

    train_imgs = [os.path.join(data_root, v, 'frames') for v in all_videos]
    train_msks = [os.path.join(data_root, v, 'segmentation') for v in all_videos]

    for v in all_videos:
        f_dir = os.path.join(data_root, v, 'frames')
        m_dir = os.path.join(data_root, v, 'segmentation')
        print(f"{v}: {len(os.listdir(f_dir))} frames, {len(os.listdir(m_dir))} masks")

    train_ds = ConcatDataset([
        SurgicalDataset(f, m, tfm=get_training_augmentations(), img_sz=img_sz)
        for f, m in zip(train_imgs, train_msks)
    ])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    model = get_deeplabv3(model_name='deeplabv3_resnet50', n_classes=n_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)['out']
            loss = loss_fn(preds, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}: Avg Training Loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, "deeplabv3_final.pth"))
    print("Training complete. Model saved to checkpoints/deeplabv3_final.pth")


if __name__ == '__main__':
    run_training()
