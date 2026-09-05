import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime
import scipy.constants as const
import iri2016.altitude as iri
from ISRpy import beampack

# Add the current directory to sys.path so we can import OXmodeCython
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from OXmodeCython import OXmode_cython

def main():
    # 1. Define location and time (Jicamarca)
    lat = -11.95
    lon = -76.87
    # 6 times at 4-hour intervals for Jan 15, 2016
    times = [datetime.datetime(2016, 1, 15, h, 0, 0) for h in range(0, 24, 4)]
    
    # 2. Define the exact altitude grid needed for the forward model
    dz = 0.5
    z = np.arange(0.01, 800, dz)
    alt_iri = np.arange(50, 800, 1.0)
    
    # 3. Compute electron gyrofrequency (fes) using IGRF/beampack for Jicamarca
    # (The magnetic field barely changes over a single day, so we compute it once)
    print("Computing magnetic field (fes) using IGRF...")
    t0 = times[0]
    jro0 = beampack.jrobeam
    igrf_year = t0.year + (t0 - datetime.datetime(t0.year, 1, 1)).total_seconds() / (
            datetime.datetime(t0.year + 1, 1, 1) - datetime.datetime(t0.year, 1, 1)).total_seconds()
            
    fes = []
    for iZ in z:
        # Evaluate the field along the zenith
        r, geoc_lat, geoc_lon, aspect, B = jro0.aspect_angle(igrf_year, jro0.xyz0 + jro0.zenith0 * iZ)
        fes.append(const.e * np.linalg.norm(B) * 1e-9 / const.m_e / 1e6 / 2 / np.pi)
    fes = np.array(fes)
    
    # 4. Setup subplots for the 6 panels
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()
    
    # 5. Loop over the times
    for i, time in enumerate(times):
        print(f"\n[{i+1}/6] Fetching IRI-2016 model at {time} UTC...")
        
        # Query IRI model
        sim = iri.IRI(time, [alt_iri[0], alt_iri[-1], 1.0], lat, lon)
        Ne_m3 = np.squeeze(sim.ne.values)
        
        # Replace negative values with NaN to avoid sqrt issues
        Ne_m3 = np.where(Ne_m3 < 0, np.nan, Ne_m3)
        
        # Compute Plasma frequency in MHz
        fps_iri = np.sqrt(Ne_m3 / 1.24e10)
        fps_iri = np.nan_to_num(fps_iri, nan=0.0) # Fill NaNs with 0
        
        # Interpolate IRI profile onto our fine grid (pad with 0 below 50 km)
        fps = np.interp(z, alt_iri, fps_iri, left=0.0, right=0.0)
        
        # Generate Predicted O and X mode traces
        print(f"[{i+1}/6] Running forward model (OXmode_cython)...")
        fvsO, vhsO, fvsX, vhsX = OXmode_cython(fps, fes, dz)
        
        # Plot on the corresponding subplot
        ax = axes[i]
        ax.plot(fps, z, 'k--', linewidth=2, label="True Profile (IRI)")
        
        if len(fvsO) > 0:
            ax.plot(fvsO, vhsO, 'r-', linewidth=2, label="Predicted O-mode")
        if len(fvsX) > 0:
            ax.plot(fvsX, vhsX, 'g-', linewidth=2, label="Predicted X-mode")
        
        ax.set_title(f"{time.strftime('%H:%M')} UTC")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Height (km)")
        
        # Scale axes
        max_f = np.nanmax(fps)
        if len(fvsX) > 0:
            max_f = max(max_f, np.nanmax(fvsX))
        ax.set_xlim(0, max(2.0, max_f * 1.1))
        ax.set_ylim(0, 800)
        ax.grid(True, linestyle=':', alpha=0.7)
        
        # Add legend only to the first subplot to save space
        if i == 0:
            ax.legend(loc="lower right")
            
    # Finalize figure
    plt.suptitle(f"Predicted Ionograms from IRI Profiles | Jicamarca, Peru | Jan 15, 2016", fontsize=16)
    plt.tight_layout()
    # plt.savefig("iri_ionogram_prediction_6panels.png")
    plt.show()

if __name__ == "__main__":
    main()
