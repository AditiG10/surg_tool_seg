import os
import sys
import csv
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

# Add parent directory to path for src imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import SurgicalDataset, get_validation_augmentations
from src.models.deeplabv3 import get_deeplabv3


def load_model(ckpt_path: str, n_classes: int, device: torch.device):
    model = get_deeplabv3(model_name='deeplabv3_resnet50', n_classes=n_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    return model


def get_val_loader(img_dir, msk_dir, img_sz=(256, 256)):
    dataset = SurgicalDataset(img_dir, msk_dir, tfm=get_validation_augmentations(), img_sz=img_sz)
    return DataLoader(dataset, batch_size=1, shuffle=False)


def compute_metrics(preds, targets, n_classes):
    preds = torch.argmax(preds, dim=1)
    tot_pix = targets.numel()
    corr_pix = (preds == targets).sum().item()

    iou_list = [0.0] * n_classes
    dice_list = [0.0] * n_classes

    for cls in range(n_classes):
        pred_msk = (preds == cls)
        true_msk = (targets == cls)

        inter = (pred_msk & true_msk).sum().item()
        uni = (pred_msk | true_msk).sum().item()
        dice_denom = pred_msk.sum().item() + true_msk.sum().item()

        if uni > 0:
            iou_list[cls] = inter / uni
        if dice_denom > 0:
            dice_list[cls] = 2 * inter / dice_denom

    mean_iou = sum(iou_list) / n_classes
    mean_dice = sum(dice_list) / n_classes
    pix_acc = corr_pix / tot_pix

    return mean_iou, mean_dice, pix_acc


def evaluate(model, loader, device, n_classes):
    model.eval()
    tot_iou, tot_dice, tot_acc, count = 0.0, 0.0, 0.0, 0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="Evaluating"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)['out']
            iou, dice, acc = compute_metrics(preds, masks, n_classes)
            tot_iou += iou
            tot_dice += dice
            tot_acc += acc
            count += 1

    return tot_iou / count, tot_dice / count, tot_acc / count


if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt_path = 'checkpoints/deeplabv3_final.pth'
    test_root = 'data/test'
    n_classes = 10
    results = []

    for i in range(41, 51):
        vid = f"video_{i:02d}"
        img_dir = os.path.join(test_root, vid, 'frames')
        msk_dir = os.path.join(test_root, vid, 'segmentation')

        if not (os.path.exists(img_dir) and os.path.exists(msk_dir)):
            print(f"Skipping {vid}: missing frames or masks.")
            continue

        print(f"Evaluating {vid}...")
        model = load_model(ckpt_path, n_classes, device)
        loader = get_val_loader(img_dir, msk_dir, img_sz=(256, 256))

        iou, dice, acc = evaluate(model, loader, device, n_classes)
        print(f"{vid} - IoU: {iou:.4f}, Dice: {dice:.4f}, Acc: {acc:.4f}")
        results.append([vid, f"{iou:.4f}", f"{dice:.4f}", f"{acc:.4f}"])

    with open("evaluation_summary.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Video", "Mean IoU", "Mean Dice", "Pixel Accuracy"])
        writer.writerows(results)

    print("Evaluation complete. Results saved to evaluation_summary.csv")
