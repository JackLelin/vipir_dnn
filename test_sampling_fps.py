import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import random

class ProfileTester:
    def fps_parabolic(self, a: np.ndarray, z: np.ndarray, beta:float=7) -> np.ndarray:
        def softplus(x):
            peak = np.max(x)
            if peak < 0.0001:
                return 0*x
            else:
                # Replaced np.log(1+np.exp(x*beta)) with np.logaddexp(0, x*beta) to prevent overflow!
                out = 1/beta * np.logaddexp(0, x*beta)
                out = out/np.max(out)*peak
                return out
                
        para1 = a[0]-((z-a[1])/(a[2]+0.01))**2
        para2 = a[3]-((z-a[4])/(a[5]+0.01))**2
        para3 = a[6]-((z-a[7])/(a[8]+0.01))**2

        fps1 = (80.6*(softplus(para1) + softplus(para2) + softplus(para3)))**0.5
        return fps1

    def fps_gauss(self, a:np.ndarray, z:np.ndarray) -> np.ndarray:
        result = a[0]*np.exp(-((z-a[1])**2)**a[6]/1000/a[2])
        result += a[3]*np.exp(-((z-a[4])**2)**a[6]/1000/a[5])
        return result

    def sampling_fps(self, z, mode):
        if mode == 0:
            a = np.array([
                np.random.uniform(0.5, 1.5) * 0.9,      # a[0]: Parabola 1 Peak Amplitude (Density)
                np.random.uniform(250, 500),            # a[1]: Parabola 1 Peak Altitude (z)
                np.random.uniform(0.65, 1.33) * 100,    # a[2]: Parabola 1 Width (Semithickness)

                np.random.uniform(0.1, 1.5) * 0.12,     # a[3]: Parabola 2 Peak Amplitude (Density)
                np.random.uniform(100, 350),            # a[4]: Parabola 2 Peak Altitude (z)
                np.random.uniform(0.65, 1.33) * 100,    # a[5]: Parabola 2 Width (Semithickness)

                np.random.uniform(0.0, 1.2) * 0.05 * (np.random.uniform(0.0, 1)>0.3),  # a[6]: Parabola 3 Peak Amplitude (Density)
                np.random.uniform(80, 140),            # a[7]: Parabola 3 Peak Altitude (z)
                np.random.uniform(0.5, 1.5) * 50,      # a[8]: Parabola 3 Width (Semithickness)
                
                np.random.uniform(5, 10)                # a[9]: Beta (Softplus smoothing parameter)
                ])
            fps = self.fps_parabolic(a, z, beta=a[-1])
        else: # mode == 1
            a = np.array([
                random.uniform(0.75, 1.20) * 10, # a[0]: Gaussian 1 Peak Amplitude
                random.uniform(300, 500),        # a[1]: Gaussian 1 Peak Altitude (z)
                random.uniform(0.5, 1.5) * 15,   # a[2]: Gaussian 1 Width
                
                random.uniform(0.8, 1.2) * 3.2,  # a[3]: Gaussian 2 Peak Amplitude
                random.uniform(100, 300),        # a[4]: Gaussian 2 Peak Altitude (z)
                random.uniform(0.5, 1.5) * 10,   # a[5]: Gaussian 2 Width
                
                random.uniform(1, 1.2)           # a[6]: Shape Factor (Exponent applied to both Gaussians)
                ])
            fps = self.fps_gauss(a, z)
        return fps, a

def main():
    tester = ProfileTester()
    z = np.linspace(0, 700, 1000)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    plt.subplots_adjust(bottom=0.15) # Leave space for the button at the bottom
    
    def draw(event=None):
        # Generate 3 parabolic and 3 gaussian random profiles
        for i in range(3):
            axes[0, i].clear()
            fps_para, a_para = tester.sampling_fps(z, mode=0)
            axes[0, i].plot(fps_para, z, 'b-', linewidth=2)
            axes[0, i].set_title(f"Parabolic Profile (Beta={a_para[-1]:.2f})")
            axes[0, i].set_xlabel("Plasma Frequency")
            axes[0, i].set_ylabel("Height (km)")
            axes[0, i].grid(True, alpha=0.3)
            axes[0, i].set_ylim(0, 700)

            axes[1, i].clear()
            fps_gauss, a_gauss = tester.sampling_fps(z, mode=1)
            axes[1, i].plot(fps_gauss, z, 'g-', linewidth=2)
            axes[1, i].set_title(f"Gaussian Profile (Shape={a_gauss[6]:.2f})")
            axes[1, i].set_xlabel("Plasma Frequency")
            axes[1, i].set_ylabel("Height (km)")
            axes[1, i].grid(True, alpha=0.3)
            axes[1, i].set_ylim(0, 700)
            
        fig.canvas.draw_idle()

    # Draw the initial batch of profiles
    draw()

    # Create the interactive button
    ax_button = plt.axes([0.4, 0.04, 0.2, 0.05])
    # We attach the button to the figure (fig.btn) to prevent it from being garbage collected!
    fig.btn = Button(ax_button, 'Quick Resample', color='lightblue', hovercolor='skyblue')
    fig.btn.on_clicked(draw)
    
    plt.show()

if __name__ == "__main__":
    main()
