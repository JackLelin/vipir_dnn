import numpy as np
import os
import glob
from tqdm import tqdm

def check_sigma_zeros():
    npy_dir = "training_data/npy"
    if not os.path.exists(npy_dir):
        print(f"Directory {npy_dir} does not exist.")
        return
        
    npy_files = glob.glob(os.path.join(npy_dir, "*.npy"))
    print(f"Found {len(npy_files)} .npy files. Scanning for sigma=0.0...")
    
    files_with_zeros = []
    
    for file in tqdm(npy_files):
        data = np.load(file, allow_pickle=True)[()]
        
        # Check O-mode
        osigma = data['osigma']
        opres = data['opres']
        
        # Check X-mode
        xsigma = data['xsigma']
        xpres = data['xpres']
        
        # Find exactly where trace is present (opres == 1) but sigma wasn't calculated (sigma == 0.0)
        o_zero_indices = np.where((opres != 0) & (osigma == 0.0))[0]
        x_zero_indices = np.where((xpres != 0) & (xsigma == 0.0))[0]
        
        if len(o_zero_indices) > 0 or len(x_zero_indices) > 0:
            files_with_zeros.append({
                'file': os.path.basename(file),
                'o_zeros': o_zero_indices,
                'x_zeros': x_zero_indices
            })
            
    print(f"\n--- Scan Complete ---")
    if len(files_with_zeros) == 0:
        print("Great news! No files contain sigma=0.0 where trace is present.")
    else:
        print(f"Found {len(files_with_zeros)} files with the zero-sigma condition!")
        print("Here is a sample of 10 affected files:")
        for info in files_with_zeros[:10]:
            print(f"\nFile: {info['file']}")
            if len(info['o_zeros']) > 0:
                print(f"  -> O-mode has 0.0 sigma at columns: {info['o_zeros']}")
            if len(info['x_zeros']) > 0:
                print(f"  -> X-mode has 0.0 sigma at columns: {info['x_zeros']}")

if __name__ == "__main__":
    check_sigma_zeros()
