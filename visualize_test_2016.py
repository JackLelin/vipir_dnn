import os
import glob
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button

from unet import UNet

# === User Configuration ===
model_path = '/mnt/LData/MLvipir_save/save_model_latest/DBCEF_VTUNET_40ktrain_GaussLabel_epoch_98_36000_loss0.4153.pth'
data_dir = '/home/jackle/Documents/MLvipir/testset2016/binary_npz'
target_size = [408, 512]
# ==========================


class BinaryIonogramDataset:
    def __init__(self, data_dir):
        self.files = sorted(glob.glob(os.path.join(data_dir, '*.npz')))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        data = np.load(path, allow_pickle=True)
        mask_full = data['mask_full']  # (512, 408)
        original = data['original'] if 'original' in data else None

        # UNet input: [1, 408, 512]
        image = torch.from_numpy(mask_full.T).float().unsqueeze(0)
        return {
            'image': image,
            'mask_full': mask_full,
            'original': original,
            'filename': os.path.basename(path),
        }


class InteractiveVisualizer:
    def __init__(self, dataset, model, device):
        self.dataset = dataset
        self.model = model
        self.device = device

        self.fig, self.axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
        plt.subplots_adjust(bottom=0.15)

        # Next button
        self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
        self.bnext = Button(self.axnext, 'Next Example')
        self.bnext.on_clicked(self.next_example)

        self.ims = {}
        self.first_plot = True

        self.next_example(None)
        plt.show()

    def next_example(self, event):
        idx = random.randint(0, len(self.dataset) - 1)
        item = self.dataset[idx]
        image = item['image']
        mask_full = item['mask_full']
        original = item['original']
        fname = item['filename']

        input_tensor = image.unsqueeze(0).to(self.device)
        with torch.no_grad():
            output = self.model(input_tensor)
        output = output.squeeze(0).cpu()

        # pred_O = torch.softmax(output[0:2, :, :], dim=0)[1, :, :].numpy().T
        # pred_X = torch.softmax(output[2:4, :, :], dim=0)[1, :, :].numpy().T
        pred_O = (output[1] - output[0]).sigmoid().numpy().T
        pred_X = (output[3] - output[2]).sigmoid().numpy().T
        pred_comb = pred_O + pred_X

        pred_comb = pred_O + pred_X

        orig_disp = np.log10(np.abs(original)) if original is not None else mask_full

        self.fig.suptitle(f"{fname} (Index: {idx})", fontsize=16)

        if self.first_plot:
            # Row 0: Original, Binary mask, and Overlays
            self.ims['orig'] = self.axs[0, 0].imshow(orig_disp, origin='lower', aspect='auto', cmap='gray')
            self.axs[0, 0].set_title('Original Ionogram')
            self.axs[0, 0].axis('off')

            self.ims['mask'] = self.axs[0, 1].imshow(mask_full, origin='lower', aspect='auto', cmap='gray')
            self.axs[0, 1].set_title('Binary Mask')
            self.axs[0, 1].axis('off')

            self.ims['over_orig_base'] = self.axs[0, 2].imshow(orig_disp, origin='lower', aspect='auto', cmap='gray')
            self.ims['over_orig_pred'] = self.axs[0, 2].imshow(pred_comb, origin='lower', aspect='auto', cmap='magma', alpha=0.55)
            self.axs[0, 2].set_title('Overlay (Original + Pred)')
            self.axs[0, 2].axis('off')

            self.ims['over_mask_base'] = self.axs[0, 3].imshow(mask_full, origin='lower', aspect='auto', cmap='gray')
            self.ims['over_mask_pred'] = self.axs[0, 3].imshow(pred_comb, origin='lower', aspect='auto', cmap='magma', alpha=0.55)
            self.axs[0, 3].set_title('Overlay (Mask + Pred)')
            self.axs[0, 3].axis('off')

            # Row 1: Model input and prediction channels
            self.ims['prob_O'] = self.axs[1, 0].imshow(pred_O, origin='lower', aspect='auto', cmap='viridis')
            self.axs[1, 0].set_title('Prob O-mode')
            self.axs[1, 0].axis('off')

            self.ims['prob_X'] = self.axs[1, 1].imshow(pred_X, origin='lower', aspect='auto', cmap='viridis')
            self.axs[1, 1].set_title('Prob X-mode')
            self.axs[1, 1].axis('off')

            self.ims['pred_O'] = self.axs[1, 2].imshow(pred_O>0.7, origin='lower', aspect='auto', cmap='viridis')
            self.axs[1, 2].set_title('Prediction O')
            self.axs[1, 2].axis('off')

            self.ims['pred_X'] = self.axs[1, 3].imshow(pred_X>0.7, origin='lower', aspect='auto', cmap='viridis')
            self.axs[1, 3].set_title('Prediction X')
            self.axs[1, 3].axis('off')

            self.first_plot = False
        else:
            self.ims['orig'].set_data(orig_disp)
            self.ims['mask'].set_data(mask_full)
            self.ims['over_orig_base'].set_data(orig_disp)
            self.ims['over_orig_pred'].set_data(pred_comb)
            self.ims['over_mask_base'].set_data(mask_full)
            self.ims['over_mask_pred'].set_data(pred_comb)

            self.ims['pred_O'].set_data(pred_O>0.7)
            self.ims['pred_X'].set_data(pred_X>0.7)
            self.ims['prob_O'].set_data(pred_O)
            self.ims['prob_X'].set_data(pred_X)

            self.fig.canvas.draw_idle()


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Loading dataset from {data_dir}...")
    dataset = BinaryIonogramDataset(data_dir)
    print(f"Found {len(dataset)} files.")

    print("Initializing model...")
    model = UNet(n_channels=1, n_classes=4, input_size=target_size, in_dropout_p=0.3, dropout_p=0.3)
    model = model.to(device)

    print(f"Loading weights from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    print("Model loaded successfully.")

    model.eval()

    print("Opening interactive visualizer...")
    InteractiveVisualizer(dataset, model, device)


if __name__ == '__main__':
    main()
