import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import datetime
from ISRpy import beampack
import scipy.constants as const
from OXmodeCython import OXmode_cython

class ProfileTester:
    def fps_parabolic(self, a: np.ndarray, z: np.ndarray, beta:float=7) -> np.ndarray:
        def softplus(x):
            peak = np.max(x)
            if peak < 0.0001:
                return 0*x
            else:
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

def main():
    tester = ProfileTester()
    dz = 0.5
    z = np.arange(0.01, 800, dz)
    
    jro0 = beampack.jrobeam
    dt0 = datetime.datetime(2016, 1, 14)
    igrf_year = dt0.year + (dt0 - datetime.datetime(dt0.year, 1, 1)).total_seconds() / (
            datetime.datetime(dt0.year + 1, 1, 1) - datetime.datetime(dt0.year, 1, 1)).total_seconds()
    
    fes = []
    for iZ in z:
        r, geoc_lat, lon, aspect, B = jro0.aspect_angle(igrf_year, jro0.xyz0 + jro0.zenith0 * iZ)
        fes.append(const.e * np.linalg.norm(B) * 1e-9 / const.m_e / 1e6 / 2 / np.pi)
    fes = np.array(fes)
    
    fig = plt.figure(figsize=(15, 8))
    # Leave room for sliders on the right
    ax_plot = fig.add_axes([0.05, 0.1, 0.45, 0.8])
    
    # Store mode in a dictionary to be mutable inside functions
    mode_state = {'mode': 0}
    
    line, = ax_plot.plot([], [], 'b-', linewidth=2, label='Plasma frequency')
    lineO, = ax_plot.plot([], [], 'r-', linewidth=2, label='O-mode')
    lineX, = ax_plot.plot([], [], 'g-', linewidth=2, label='X-mode')
    ax_plot.set_xlabel("Frequency [MHz]")
    ax_plot.set_ylabel("Height [km]")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_ylim(0, 800)
    ax_plot.legend()
    
    # Define Sliders Configuration
    # Parabolic parameters (Mode 0)
    para_labels = [
        "P1 Amp", "P1 Alt", "P1 Width",
        "P2 Amp", "P2 Alt", "P2 Width",
        "P3 Amp", "P3 Alt", "P3 Width",
        "Beta"
    ]
    a_para_init = [0.9, 375.0, 100.0, 0.22, 261, 144, 0.042, 130., 40, 7.5]
    para_mins = [0.0, 100.0, 10.0, 0.0, 50.0, 10.0, 0.0, 50.0, 10.0, 1.0]
    para_maxs = [2.0, 700.0, 200.0, 0.5, 500.0, 200.0, 0.2, 200.0, 150.0, 20.0]
    
    # Gaussian parameters (Mode 1)
    gauss_labels = [
        "G1 Amp", "G1 Alt", "G1 Width",
        "G2 Amp", "G2 Alt", "G2 Width",
        "Shape"
    ]
    a_gauss_init = [10., 409, 20.8, 3.2, 200.0, 7.4, 1.02]
    gauss_mins = [0.0, 100.0, 1.0, 0.0, 50.0, 1.0, 0.5]
    gauss_maxs = [20.0, 700.0, 50.0, 10.0, 500.0, 50.0, 3.0]
    
    para_slider_left = 0.55
    gauss_slider_left = 0.78
    slider_width = 0.18
    slider_height = 0.03
    slider_bottom = 0.85
    slider_spacing = 0.05
    
    para_slider_axes = []
    para_sliders = []
    gauss_slider_axes = []
    gauss_sliders = []
    
    for i in range(10):
        ax_s = fig.add_axes([para_slider_left, slider_bottom - i*slider_spacing, slider_width, slider_height])
        s = Slider(ax_s, para_labels[i], para_mins[i], para_maxs[i], valinit=a_para_init[i])
        para_slider_axes.append(ax_s)
        para_sliders.append(s)
        
    for i in range(7):
        ax_s = fig.add_axes([gauss_slider_left, slider_bottom - i*slider_spacing, slider_width, slider_height])
        s = Slider(ax_s, gauss_labels[i], gauss_mins[i], gauss_maxs[i], valinit=a_gauss_init[i])
        gauss_slider_axes.append(ax_s)
        gauss_sliders.append(s)
        
    fig.sampled_lines = []
    
    def update(val=None, a_override=None):
        for sl in fig.sampled_lines:
            try:
                sl.remove()
            except Exception:
                pass
        fig.sampled_lines.clear()

        if mode_state['mode'] in [0, 2]:
            if a_override is not None:
                a = a_override.copy()
                for s, v in zip(para_sliders, a):
                    l = s.ax.axvline(v, color='red', linewidth=2, alpha=0.7)
                    fig.sampled_lines.append(l)
            else:
                a = np.array([s.val for s in para_sliders])
                
            if mode_state['mode'] == 2:
                a[3] = 0.0
                a[6] = 0.0
                
            fps = tester.fps_parabolic(a, z, beta=a[-1])
            line.set_data(fps, z)
            line.set_color('blue')
            if mode_state['mode'] == 2:
                title_text = f"Single Parabola Profile (Beta={a[-1]:.2f})"
            else:
                title_text = f"Parabolic Profile (Beta={a[-1]:.2f})"
            if a_override is not None:
                title_text += " [PERTURBED]"
            ax_plot.set_title(title_text)
        else:
            if a_override is not None:
                a = a_override
                for s, v in zip(gauss_sliders, a):
                    l = s.ax.axvline(v, color='red', linewidth=2, alpha=0.7)
                    fig.sampled_lines.append(l)
            else:
                a = np.array([s.val for s in gauss_sliders])
            fps = tester.fps_gauss(a, z)
            line.set_data(fps, z)
            line.set_color('green')
            title_text = f"Gaussian Profile (Shape={a[6]:.2f})"
            if a_override is not None:
                title_text += " [PERTURBED]"
            ax_plot.set_title(title_text)
            
        fvsO, vhsO, fvsX, vhsX = OXmode_cython(fps, fes, dz)
        lineO.set_data(fvsO, vhsO)
        lineX.set_data(fvsX, vhsX)
            
        # Dynamically adjust the x-axis to fit the profile
        max_f = np.nanmax(fps)
        if len(fvsX) > 0:
            max_f = max(max_f, np.nanmax(fvsX))
        ax_plot.set_xlim(0, max(1.0, max_f * 1.1) if not np.isnan(max_f) else 15)
        fig.canvas.draw_idle()
        
    # Attach update function to sliders
    for s in para_sliders + gauss_sliders:
        s.on_changed(update)
        
    # Button to switch modes
    ax_button = fig.add_axes([0.55, 0.15, 0.41, 0.04])
    btn_switch = Button(ax_button, 'Mode: Parabolic', color='lightblue', hovercolor='skyblue')
    
    ax_btn_sample = fig.add_axes([0.55, 0.10, 0.20, 0.04])
    btn_sample = Button(ax_btn_sample, 'Sample', color='lightgoldenrodyellow', hovercolor='khaki')
    
    ax_btn_save = fig.add_axes([0.76, 0.10, 0.20, 0.04])
    btn_save = Button(ax_btn_save, 'Save', color='lightpink', hovercolor='pink')
    
    def switch_mode(event):
        if mode_state['mode'] == 0:
            mode_state['mode'] = 1
            btn_switch.label.set_text('Mode: Gaussian')
            btn_switch.color = 'lightgreen'
            btn_switch.hovercolor = 'palegreen'
            for s in para_sliders:
                s.poly.set_alpha(0.2)
                s.valtext.set_alpha(0.2)
                s.label.set_alpha(0.2)
            for s in gauss_sliders:
                s.poly.set_alpha(1.0)
                s.valtext.set_alpha(1.0)
                s.label.set_alpha(1.0)
        elif mode_state['mode'] == 1:
            mode_state['mode'] = 2
            btn_switch.label.set_text('Mode: Single Parabola')
            btn_switch.color = 'lightcyan'
            btn_switch.hovercolor = 'cyan'
            for i, s in enumerate(para_sliders):
                alpha = 0.2 if i in [3, 4, 5, 6, 7, 8] else 1.0
                s.poly.set_alpha(alpha)
                s.valtext.set_alpha(alpha)
                s.label.set_alpha(alpha)
            for s in gauss_sliders:
                s.poly.set_alpha(0.2)
                s.valtext.set_alpha(0.2)
                s.label.set_alpha(0.2)
        else:
            mode_state['mode'] = 0
            btn_switch.label.set_text('Mode: Parabolic')
            btn_switch.color = 'lightblue'
            btn_switch.hovercolor = 'skyblue'
            for s in para_sliders:
                s.poly.set_alpha(1.0)
                s.valtext.set_alpha(1.0)
                s.label.set_alpha(1.0)
            for s in gauss_sliders:
                s.poly.set_alpha(0.2)
                s.valtext.set_alpha(0.2)
                s.label.set_alpha(0.2)
        fig.canvas.draw_idle()
        update()
        
    # Set initial alpha
    for s in gauss_sliders:
        s.poly.set_alpha(0.2)
        s.valtext.set_alpha(0.2)
        s.label.set_alpha(0.2)
        
    btn_switch.on_clicked(switch_mode)
    
    # Store references to widgets to prevent garbage collection
    fig.widgets = {
        'para_sliders': para_sliders,
        'gauss_sliders': gauss_sliders,
        'btn_switch': btn_switch,
        'btn_sample': btn_sample,
        'btn_save': btn_save
    }
    
    def sample_perturbed(event):
        import random
        if mode_state['mode'] == 0:
            v = np.array([s.val for s in para_sliders])
            a = np.array([
                np.random.uniform(0.5, 1.5) * v[0],
                np.random.uniform(0.8, 1.2) * v[1],
                np.random.uniform(0.65, 1.33) * v[2],
                np.random.uniform(0.66, 1.5) * v[3],
                np.random.uniform(0.8, 1.2) * v[4],
                np.random.uniform(0.65, 1.33) * v[5],
                np.random.uniform(0.6, 1.2) * v[6] * (1 if np.random.uniform(0, 1)>0.3 else 0),
                np.random.uniform(0.8, 1.2) * v[7],
                np.random.uniform(0.5, 1.3) * v[8],
                np.random.uniform(0.8, 1.2) * v[9]
            ])
            update(a_override=a)
        elif mode_state['mode'] == 2:
            v = np.array([s.val for s in para_sliders])
            a = np.array([
                np.random.uniform(0.3, 1.8) * v[0],
                np.random.uniform(0.8, 1.2) * v[1],
                np.random.uniform(0.65, 1.33) * v[2],
                0.0,
                v[4],
                v[5],
                0.0,
                v[7],
                v[8],
                np.random.uniform(0.8, 1.2) * v[9]
            ])
            update(a_override=a)
        else:
            v = np.array([s.val for s in gauss_sliders])
            a = np.array([
                random.uniform(0.75, 1.20) * v[0],
                random.uniform(0.8, 1.2) * v[1],
                random.uniform(0.5, 1.5) * v[2],
                random.uniform(0.8, 1.2) * v[3],
                random.uniform(0.8, 1.2) * v[4],
                random.uniform(0.5, 1.5) * v[5],
                random.uniform(1, 1.2) * v[6]
            ])
            update(a_override=a)
            
    btn_sample.on_clicked(sample_perturbed)
    
    def save_values(event):
        if mode_state['mode'] == 0:
            a = np.array([s.val for s in para_sliders])
            labels = para_labels
            filename = "saved_parabolic_params.txt"
        elif mode_state['mode'] == 2:
            a = np.array([s.val for s in para_sliders])
            a[3] = 0.0
            a[6] = 0.0
            labels = para_labels
            filename = "saved_single_parabola_params.txt"
        else:
            a = np.array([s.val for s in gauss_sliders])
            labels = gauss_labels
            filename = "saved_gaussian_params.txt"
            
        with open(filename, 'a') as f:
            f.write(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
            for label, val in zip(labels, a):
                f.write(f"{label}: {val}\n")
            f.write("\n")
        print(f"Saved parameters to {filename}")
        
    btn_save.on_clicked(save_values)
    
    # Initial draw
    update()
    plt.show()

if __name__ == "__main__":
    main()
