import numpy as np
import matplotlib.pyplot as plt

def test_smooth_relu():
    # A dummy x-axis for height (z)
    z = np.linspace(100, 600, 1000)
    
    # Parameters for 3 parabolas (simulating E, F1, F2 layers)
    # The first parabola (E layer) will be very small to show the issue!
    a = np.array([
        0.01, 150, 20,   # Parabola 1: tiny peak of 0.01 at z=150, width 20
        5.0,  300, 30,   # Parabola 2: peak of 5.0 at z=300, width 80
        9.0,  450, 30   # Parabola 3: peak of 9.0 at z=450, width 100
    ])

    para1 = a[0] - ((z - a[1]) / (a[2] + 0.01))**2
    para2 = a[3] - ((z - a[4]) / (a[5] + 0.01))**2
    para3 = a[6] - ((z - a[7]) / (a[8] + 0.01))**2

    # 1. Standard ReLU (Sharp)
    relu_total = np.maximum(para1, 0) + np.maximum(para2, 0) + np.maximum(para3, 0)
    
    # 2. Old Softplus
    beta = 2
    old_sp_1 = 1/beta * np.logaddexp(0, para1 * beta)
    old_sp_2 = 1/beta * np.logaddexp(0, para2 * beta)
    old_sp_3 = 1/beta * np.logaddexp(0, para3 * beta)
    softplus_total = old_sp_1 + old_sp_2 + old_sp_3
    
    # 3. Pseudo-Huber Smooth ReLU (Smooth and Accurate)
    epsilon = 0.01
    def smooth_relu(x):
        return 0.5 * (np.sqrt(x**2 + epsilon**2) + x)
        
    smooth_total = smooth_relu(para1) + smooth_relu(para2) + smooth_relu(para3)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: The full view
    ax1.plot(relu_total, z, 'r-', label='ReLU (Sharp, Accurate)', linewidth=1, alpha=0.6)
    ax1.plot(softplus_total, z, 'g-', label=f'Softplus (beta={beta})', linewidth=1)
    ax1.plot(smooth_total, z, 'b--', label=f'Smooth ReLU ($\epsilon={epsilon}$)', linewidth=1)
    ax1.set_title("Full Profile: 3 Parabolas Combined")
    ax1.set_xlabel("Electron Density")
    ax1.set_ylabel("Height (z)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Zoomed in on the tiny first parabola
    ax2.plot(relu_total, z, 'r-', label='ReLU', linewidth=1, alpha=0.6)
    ax2.plot(softplus_total, z, 'g-', label='Softplus', linewidth=1)
    ax2.plot(smooth_total, z, 'b--', label='Smooth ReLU', linewidth=1)
    ax2.set_title("Zoomed View: The Tiny Parabola (Peak=0.01)")
    ax2.set_xlabel("Electron Density")
    ax2.set_ylabel("Height (z)")
    ax2.set_xlim(-0.01, 0.15)
    ax2.set_ylim(100, 200)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # plt.savefig('test_smooth_relu.png', dpi=150)
    print("Saved plot to 'test_smooth_relu.png'!")
    plt.show()

if __name__ == "__main__":
    test_smooth_relu()
