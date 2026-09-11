import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from unet import UNet
from testionogramdataset import HandPickTraceDataset

# === User Configuration ===
# Enter the absolute or relative path to the trained model checkpoint here
model_path = '/mnt/LData/MLvipir_save/save_model_latest/DBCEF_VTUNET_40ktrain_GaussLabel_epoch_116_36000_loss0.4269.pth' 
num_examples = 5
# ==========================

def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    target_size = [408, 512]
    resize = 'pad'

    print("Loading test dataset...")
    # Using the exact same dataset class and setup as in evaluate.py / model_train.py
    test_dataset = HandPickTraceDataset('VIPIR_PROC/labeled', target_size=target_size, resize=resize)

    # Pick 5 random indices from the test set
    indices = random.sample(range(len(test_dataset)), min(num_examples, len(test_dataset)))

    print("Initializing model...")
    model = UNet(n_channels=1, n_classes=4, input_size=target_size, in_dropout_p=0.3, dropout_p=0.3)
    model = model.to(device)

    if "YOUR_MODEL_FILE_HERE.pth" not in model_path:
        try:
            print(f"Loading weights from {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    else:
        print("WARNING: 'model_path' not set. Using UNTRAINED weights.")

    model.eval()

    from matplotlib.widgets import Button

    class InteractiveVisualizer:
        def __init__(self, dataset, model, device):
            self.dataset = dataset
            self.model = model
            self.device = device
            
            self.fig, self.axs = plt.subplots(ncols=4, nrows=2, figsize=(20, 10))
            plt.subplots_adjust(bottom=0.15)
            
            # Setup button
            self.axnext = plt.axes([0.81, 0.05, 0.1, 0.075])
            self.bnext = Button(self.axnext, 'Next Example')
            self.bnext.on_clicked(self.next_example)
            
            self.ims = {}
            self.first_plot = True
            
            self.next_example(None)
            plt.show()
            
        def next_example(self, event):
            idx = random.randint(0, len(self.dataset) - 1)
            data = self.dataset[idx]
            image, gt = data['image'], data['mask']
            
            input_tensor = image.unsqueeze(0).to(self.device)
            with torch.no_grad():
                output = self.model(input_tensor)
            output = output.squeeze(0).cpu()
            
            # pred_O = torch.argmax(output[0:2, :, :], dim=0).numpy()
            # pred_X = torch.argmax(output[2:4, :, :], dim=0).numpy()
            pred_O = output[1, :, :].numpy()
            pred_X = output[3, :, :].numpy()

            gt_O = gt[0, :, :].numpy()
            gt_X = gt[1, :, :].numpy()
            input_img = image[0].numpy()
            
            self.fig.suptitle(f"Test Example (Dataset Index: {idx})", fontsize=16)
            
            if self.first_plot:
                self.ims['gt_in'] = self.axs[0, 0].imshow(input_img.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
                self.axs[0, 0].set_title('Input Image')
                self.axs[0, 0].axis('off')
                
                self.ims['gt_O'] = self.axs[0, 1].imshow(gt_O.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[0, 1].set_title('Ground Truth O-mode')
                self.axs[0, 1].axis('off')
                
                self.ims['gt_X'] = self.axs[0, 2].imshow(gt_X.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[0, 2].set_title('Ground Truth X-mode')
                self.axs[0, 2].axis('off')
                
                self.ims['gt_comb'] = self.axs[0, 3].imshow(gt_O.T + gt_X.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[0, 3].set_title('Ground Truth Combined')
                # self.axs[0, 3].axis('off')
                
                self.ims['pred_in'] = self.axs[1, 0].imshow(input_img.T, origin='lower', aspect='auto', cmap='gray', interpolation='none')
                self.axs[1, 0].set_title('Input Image')
                self.axs[1, 0].axis('off')
                
                self.ims['pred_O'] = self.axs[1, 1].imshow(pred_O.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[1, 1].set_title('Prediction O-mode')
                self.axs[1, 1].axis('off')
                
                self.ims['pred_X'] = self.axs[1, 2].imshow(pred_X.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[1, 2].set_title('Prediction X-mode')
                self.axs[1, 2].axis('off')
                
                self.ims['pred_comb'] = self.axs[1, 3].imshow(pred_O.T + pred_X.T, origin='lower', aspect='auto', cmap='viridis', interpolation='none')
                self.axs[1, 3].set_title('Prediction Combined')
                self.axs[1, 3].axis('off')
                
                self.first_plot = False
            else:
                self.ims['gt_in'].set_data(input_img.T)
                self.ims['gt_O'].set_data(gt_O.T)
                self.ims['gt_X'].set_data(gt_X.T)
                self.ims['gt_comb'].set_data(gt_O.T + gt_X.T)
                
                self.ims['pred_in'].set_data(input_img.T)
                self.ims['pred_O'].set_data(pred_O.T)
                self.ims['pred_X'].set_data(pred_X.T)
                self.ims['pred_comb'].set_data(pred_O.T + pred_X.T)
                self.fig.canvas.draw_idle()

    print("Opening interactive visualizer...")
    vis = InteractiveVisualizer(test_dataset, model, device)

    print("Done!")

if __name__ == '__main__':
    main()
