import random
import numpy as np
import glob
from skimage import morphology
from scipy.sparse import coo_matrix
from scipy.signal import find_peaks
from PIL import Image
import matplotlib.pyplot as plt

from OXmodeCython import OXmode_cython

class GenTrain():
    def __init__(self, freqs, hts, ipp, training_dir, noise_dir):
        self.freqs = freqs
        self.hts = hts
        self.ipp = ipp

        self.training_dir = training_dir
        self.noise_dir = 'VIPIR_PROC/Dog_noise/'
        self.noise_files = glob.glob1(noise_dir,'*.npy')

    def fps_parabolic(self, a: np.ndarray, z: np.ndarray, beta:float=7) -> np.ndarray:
        # print(a)
        def softplus(x):
            peak = np.max(x)
            if peak < 0.0001:
                return 0*x
            else:
                out = 1/beta*np.log(1+np.exp(x*beta))
                out = out/np.max(out)*peak
                return out
        para1 = a[0]-((z-a[1])/(a[2]+0.01))**2
        para2 = a[3]-((z-a[4])/(a[5]+0.01))**2
        para3 = a[6]-((z-a[7])/(a[8]+0.01))**2

        # temp = np.vstack([para1,para2,para3])
        # fps1 = np.clip(np.sum(temp*softmax(temp,axis=0),axis=0), 0, 100)**0.5
        fps1 = (80.6*(softplus(para1) + softplus(para2) + softplus(para3)))**0.5
        # fps1 = (80.6*np.max(np.vstack([para1,para2,para3,np.zeros_like(para1)]), axis=0))**0.5
        return fps1#,fps2

    def fps_gauss(self, a:np.ndarray, z:np.ndarray) -> np.ndarray:
        result = a[0]*np.exp(-((z-a[1])**2)**a[6]/1000/a[2])
        result += a[3]*np.exp(-((z-a[4])**2)**a[6]/1000/a[5])
        return result

    def fps_chapman(self, a: np.ndarray, z: np.ndarray) -> np.ndarray:
        def chapman_fp(z, fp_peak, hm, H):
            if fp_peak == 0:
                return np.zeros_like(z)
            normalized_z = (z - hm) / (H + 0.01)
            return fp_peak * np.exp(0.25 * (1 - normalized_z - np.exp(-normalized_z)))
            
        result = chapman_fp(z, a[0], a[1], a[2])
        result += chapman_fp(z, a[3], a[4], a[5])
        if len(a) > 6:
            result += chapman_fp(z, a[6], a[7], a[8])
        return result

    def fps_iri(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        import iri2016.altitude as iri
        import datetime
        
        alt_iri = np.arange(50, 800, 1.0)
        
        while True:
            year = np.random.randint(2010, 2021)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 28)
            hour = np.random.randint(0, 24)
            minute = np.random.choice([0, 15, 30, 45])
            time = datetime.datetime(year, month, day, hour, minute, 0)
            
            lat = np.random.uniform(-60, 60)
            lon = np.random.uniform(-180, 180)
            
            try:
                sim = iri.IRI(time, [alt_iri[0], alt_iri[-1], 1.0], lat, lon)
                Ne_m3 = np.squeeze(sim.ne.values)
                Ne_m3 = np.where(Ne_m3 < 0, 0.0, Ne_m3)
                fps_iri_vals = np.sqrt(Ne_m3 / 1.24e10)
                fps_iri_vals = np.nan_to_num(fps_iri_vals, nan=0.0)
                fps = np.interp(z, alt_iri, fps_iri_vals, left=0.0, right=0.0)
                
                max_fps = np.max(fps)
                # Ensure the ionosphere has a valid shape before scaling
                if max_fps > 0.5:
                    # Scale the profile to guarantee a strong ionosphere (peak between 6 and 12 MHz)
                    # This guarantees OXmode_cython will find reflections and fvs won't be empty!
                    fps = fps * (np.random.uniform(6.0, 12.0) / max_fps)
                    a = np.array([lat, lon, year, month, day, hour, minute])
                    return fps, a
            except Exception:
                # If IRI fails (e.g. year out of bounds for data), try again
                continue

    def sampling_fps(self, z, mode): # sampling_fps(self, z:np.ndarray, mode:int) -> tuple[np.ndarray, np.ndarray]:
        # mode = 0
        if mode == 0:
            a = np.array([
                np.random.uniform(0.5, 1.5) * 0.9,      # a[0]: Parabola 1 Peak Amplitude (Density)
                np.random.uniform(0.8, 1.2) * 375,          # a[1]: Parabola 1 Peak Altitude (z)
                np.random.uniform(0.65, 1.33) * 100,    # a[2]: Parabola 1 Width (Semithickness)

                np.random.uniform(0.66, 1.5) * 0.22,     # a[3]: Parabola 2 Peak Amplitude (Density)
                np.random.uniform(0.8, 1.2) * 261,            # a[4]: Parabola 2 Peak Altitude (z)
                np.random.uniform(0.65, 1.33) * 144,    # a[5]: Parabola 2 Width (Semithickness)

                np.random.uniform(0.6, 1.2) * 0.045 * (np.random.uniform(0.0, 1)>0.3),  # a[6]: Parabola 3 Peak Amplitude (Density)
                np.random.uniform(0.8, 1.2) * 130,         # a[7]: Parabola 3 Peak Altitude (z)
                np.random.uniform(0.8, 1.2) * 40,      # a[8]: Parabola 3 Width (Semithickness)
                
                np.random.uniform(5, 10)                # a[9]: Beta (Softplus smoothing parameter)
                ])
            fps = self.fps_parabolic(a,z,beta=a[-1])
        
        elif mode == 2:
            a = np.array([
                np.random.uniform(0.3, 1.8) * 0.4,      # a[0]: Parabola 1 Peak Amplitude (Density)
                np.random.uniform(220, 400), # 375           # a[1]: Parabola 1 Peak Altitude (z)
                np.random.uniform(0.65, 1.33) * 140,    # a[2]: Parabola 1 Width (Semithickness)
                0,     # a[3]: Parabola 2 Peak Amplitude (Density)
                250,            # a[4]: Parabola 2 Peak Altitude (z)
                50,    # a[5]: Parabola 2 Width (Semithickness)
                0,  # a[6]: Parabola 3 Peak Amplitude (Density)
                100,   # 130         # a[7]: Parabola 3 Peak Altitude (z)
                50,      # a[8]: Parabola 3 Width (Semithickness)
                np.random.uniform(5, 10)                # a[9]: Beta (Softplus smoothing parameter)
                ])
            fps = self.fps_parabolic(a,z,beta=a[-1])
            # foE = fps[np.argmin(np.abs(a[4] - z))]
        elif mode == 1:
            a = np.array([
                random.uniform(0.75, 1.20) * 10, # a[0]: Gaussian 1 Peak Amplitude
                random.uniform(300, 500),  #409      # a[1]: Gaussian 1 Peak Altitude (z)
                random.uniform(0.5, 1.5) * 20.8, # a[2]: Gaussian 1 Width
                
                random.uniform(0.8, 1.2) * 3.2,  # a[3]: Gaussian 2 Peak Amplitude
                random.uniform(100, 300),  #200      # a[4]: Gaussian 2 Peak Altitude (z)
                random.uniform(0.5, 1.5) * 7.4,   # a[5]: Gaussian 2 Width
                
                random.uniform(1, 1.2)         # a[6]: Shape Factor (Exponent applied to both Gaussians)
                ])
            fps = self.fps_gauss(a,z)
            # foE = fps[np.argmin(np.abs(a[4] - z))]
        elif mode == 3: # Chapman
            a = np.array([
                np.random.uniform(0.5, 1.5) * 8,        # a[0]: F2 Peak fp (MHz)
                np.random.uniform(220, 400),            # a[1]: F2 Peak Altitude (hm)
                np.random.uniform(30, 80),              # a[2]: F2 Scale Height (H)
                
                np.random.uniform(0.2, 1.0) * 3,        # a[3]: E Peak fp (MHz)
                np.random.uniform(90, 130),             # a[4]: E Peak Altitude (hm)
                np.random.uniform(5, 15),               # a[5]: E Scale Height (H)
                
                np.random.uniform(0.1, 0.5) * 4 * (np.random.uniform(0.0, 1)>0.3),  # a[6]: F1 Peak fp (MHz)
                np.random.uniform(150, 220),            # a[7]: F1 Peak Altitude (hm)
                np.random.uniform(15, 30),              # a[8]: F1 Scale Height (H)
                ])
            fps = self.fps_chapman(a, z)
        elif mode == 4: # IRI
            fps, a = self.fps_iri(z)
        else:
            fps = np.zeros_like(z)
            a = np.array([])
        return fps,a

    def Coo2Mat(self, fvs:np.ndarray, vhs:np.ndarray, vh_start = 40, fs_start = None) -> np.ndarray:
        # Turning the output pairs from forward model to continuous traces stored in matrix
        trace_mat = np.zeros([self.hts.shape[0],self.freqs.shape[0]])
        if vhs.shape[0] == 0 or fvs.shape[0] == 0:
            return trace_mat
        vh_start *= np.random.uniform(0.9,1.1)
        fs_start = self.freqs[0] if fs_start is None else fs_start
        i_start = np.max([np.argmin(np.abs(fvs - fs_start)), np.argmin(np.abs(vhs-vh_start))])
        ivh_prev, ifv_prev = np.argmin(np.abs(self.hts - vhs[i_start])), np.argmin(np.abs(self.freqs - fvs[i_start]))
        for (vh,fv) in zip(vhs[i_start+1:],fvs[i_start+1:]):
            if fv <= self.freqs.max() and fv >= self.freqs.min() and vh <= self.hts.max() and vh >= self.hts.min():
                ivh, ifv = np.argmin(np.abs(self.hts - vh)), np.argmin(np.abs(self.freqs - fv))
                n = max(abs(ifv_prev - ifv),abs(ivh_prev - ivh))+1
                if vh > vh_start and fv>fs_start:
                    ivs = np.rint(np.linspace(ivh_prev,ivh,n)).astype(np.int_)
                    ifs = np.rint(np.linspace(ifv_prev,ifv,n)).astype(np.int_)
                    trace_mat[ivs, ifs] = 1
                ifv_prev,ivh_prev = ifv,ivh
        return trace_mat

    def scatter_around_points(self, x_data : np.ndarray, y_data : np.ndarray, sigma_x, sigma_y, points=5):
        # Widen the trace
        M, N = self.freqs.shape[0], self.hts.shape[0]
        points = np.random.randint(3,5+1)
        
        if x_data.shape[0] == 0:
            return np.array([]), np.array([])
            
        # 1. Generate all random offsets at once and add the base coordinates via broadcasting
        nx = np.rint(x_data[:, None] + np.random.uniform(-sigma_x, sigma_x, (x_data.shape[0], points))).flatten()
        ny = np.rint(y_data[:, None] + np.random.uniform(-sigma_y, sigma_y, (y_data.shape[0], points))).flatten()
        
        # 2. Vectorized bounds check across the entire flattened array
        valid = (ny >= 0) & (ny < N) & (nx >= 0) & (nx < M)
        
        return nx[valid], ny[valid]

    def creating_multihop(self, fvs, vhs, fmin = 3):
        outmat = 0
        fmin = self.freqs[0] if fmin is None else fmin
        fmin_idx = np.argmin(np.abs(fvs-fmin))
        delta_vhs = np.abs(vhs[fmin_idx:]-vhs[fmin_idx-1:-1])
        fvs = np.copy(fvs[fmin_idx:])
        vhs = np.copy(vhs[fmin_idx:])
        indices = delta_vhs < np.random.randint(3,11)
        for i in range(2,np.random.randint(3,5)):
            vhs2 = vhs*i
            for j in range(3):
                random_fs_start = self.freqs[0] if np.random.uniform(0, 1) < 0.5 else fvs[0]
                mh_trace = self.Coo2Mat(fvs[indices],(vhs2-self.ipp*j)[indices],fs_start=random_fs_start)
                ivhs_mh, ifvs_mh = np.nonzero(mh_trace)
                # Generate noisy data
                sigma_x = np.random.uniform(1.8,2.2)*i/1.5
                sigma_y = np.random.uniform(1.8,2.2)*i/1.5
                fvs_mh_noisy_x, vhs_mh_noisy_y = self.scatter_around_points(np.array(ifvs_mh), np.array(ivhs_mh), sigma_x=sigma_x, sigma_y=sigma_y)
                outmat = outmat + coo_matrix((np.ones(fvs_mh_noisy_x.shape), (vhs_mh_noisy_y,fvs_mh_noisy_x)), shape=(self.hts.shape[0], self.freqs.shape[0])).toarray()
        return outmat

    def adding_holes_in_trace(self, noise_map, fill_rate = 0.01):
        N = int(noise_map.shape[0] * noise_map.shape[1] * fill_rate)
        loc_x = np.random.randint(0,noise_map.shape[1], N)
        loc_y = np.random.randint(0,noise_map.shape[0], N)
        hole_map = coo_matrix((np.ones(N, dtype=np.int_), (loc_y, loc_x)), shape=noise_map.shape).toarray()
        hole_map = 1 - morphology.dilation(hole_map, np.array([[0,1,0],[1,1,1],[0,1,0]]))
        noise_map *= hole_map

    def add_noise_lines(self, inmat):
        # add irregular short noise lines 
        line_mat = np.zeros([self.hts.shape[0],self.freqs.shape[0]])
        M = self.freqs.shape[0]
        N = self.hts.shape[0]
        for _ in range(80):
            ifv_start = np.random.randint(0, M)
            ivh_start = np.random.randint(0, N)
            ifv_end = ifv_start + np.random.randint(-3, 3)
            ivh_end = ivh_start + np.random.randint(10,25)
            if ifv_end < M and ifv_end >= 0 and ivh_end < N and ivh_end >= 0:
                n = max(abs(ifv_end - ifv_start),abs(ivh_end - ivh_start))+1
                ivs = np.rint(np.linspace(ivh_start,ivh_end,n)).astype(np.int_)
                ifs = np.rint(np.linspace(ifv_start,ifv_end,n)).astype(np.int_)
                line_mat[ivs, ifs] = 1
        ivhs, ifvs = np.nonzero(line_mat)
        sigma_x = np.random.uniform(1,1.8)
        sigma_y = np.random.uniform(1,1.8)
        fvs_thick_x, vhs_thick_y = self.scatter_around_points(np.array(ifvs), np.array(ivhs), sigma_x=sigma_x, sigma_y=sigma_y)
        outmat = coo_matrix((np.ones(fvs_thick_x.shape), (vhs_thick_y,fvs_thick_x)), shape=(self.hts.shape[0], self.freqs.shape[0])).toarray()
        outmat = (outmat + inmat) > 0
        return outmat.astype(np.int_)

    def add_horizontal_noise_lines(self, inmat):
        # Creates broken, dashed horizontal lines typical of RF broadcast interference
        line_mat = np.zeros([self.hts.shape[0], self.freqs.shape[0]])
        M = self.freqs.shape[0]
        N = self.hts.shape[0]
        
        band_heights = []
        
        # High probability (80%) for the relatively fixed primary band
        if np.random.rand() < 0.8:
            band_heights.append(np.random.randint(N // 5, N // 3))
            
        # Lower probability (40%) for the moving, random bands
        if np.random.rand() < 0.4:
            for _ in range(np.random.randint(1, 3)): # Create 1 to 2 random bands
                band_heights.append(np.random.randint(0, N))
        
        for ivh in band_heights:
            
            # Start somewhere in the left third of the image
            current_f = np.random.randint(0, M // 3)
            while current_f < M:
                # Length of the visible dash (Morse-code like)
                dash_len = np.random.randint(5, 40)
                end_f = min(current_f + dash_len, M)
                
                # Fill the dash center line
                line_mat[ivh, current_f:end_f] = 1
                
                # Gap before the next dash
                gap_len = np.random.randint(3, 15)
                current_f = end_f + gap_len
                
        ivhs, ifvs = np.nonzero(line_mat)
        if len(ivhs) > 0:
            sigma_x = np.random.uniform(1, 5)
            sigma_y = np.random.uniform(3, 10) # Gaussian spread for band width
            points = np.random.randint(15, 30) # Higher density to form the band from a 1D line
            
            # Gaussian scatter in Y for dense center / sparse edges, Uniform scatter in X
            fvs_thick_x = np.rint(ifvs[:, None] + np.random.uniform(-sigma_x, sigma_x, (ifvs.shape[0], points))).flatten()
            vhs_thick_y = np.rint(ivhs[:, None] + np.random.normal(0, sigma_y, (ivhs.shape[0], points))).flatten()
            
            # Filter bounds
            valid = (vhs_thick_y >= 0) & (vhs_thick_y < N) & (fvs_thick_x >= 0) & (fvs_thick_x < M)
            fvs_thick_x = fvs_thick_x[valid]
            vhs_thick_y = vhs_thick_y[valid]
            
            outmat = coo_matrix((np.ones(fvs_thick_x.shape), (vhs_thick_y, fvs_thick_x)), shape=(N, M)).toarray()
            outmat = (outmat + inmat) > 0
            return outmat.astype(np.int_)
        return inmat


    def gen_noise_blocks(self):
        M = self.freqs.shape[0]
        N = self.hts.shape[0]
        noise_x = []
        noise_y = []
        if np.random.uniform(0,1) < 0.4:
            for _ in range(np.random.randint(20,45)):
                r = np.random.uniform(0,1)
                if r < 0.5: # 50% small blocks
                    size_x = np.random.randint(2,5)
                    size_y = np.random.randint(4,7)
                elif r < 0.6: # 10% long blocks
                    size_x = np.random.randint(5,10)  
                    size_y = np.random.randint(30,50)
                else: # 40% wide blocks
                    size_x = np.random.randint(30,50)
                    size_y = np.random.randint(7,15)                        
                loc_x = np.random.randint(0,M-1)
                loc_y = np.random.randint(0,N-1)
                n = int(size_x * size_y * np.random.uniform(0.3,0.9))
                noise_x += (loc_x + np.random.uniform(-size_x/2,size_x/2,n)).astype(np.int_).tolist()
                noise_y += (loc_y + np.random.uniform(-size_y/2,size_y/2,n)).astype(np.int_).tolist()
        else :
            percent = 0.7
            noise_list = np.load(self.noise_dir + self.noise_files[np.random.randint(0, len(self.noise_files)-1)], allow_pickle=True)[()]['noise_list']
            random.shuffle(noise_list)
            delta_d = 100
            for cord_x, cord_y in noise_list[:int(len(noise_list) * percent)]:
                dx = np.random.randint(-delta_d,delta_d)
                dy = np.random.randint(-delta_d,delta_d)
                noise_x += (cord_x+dx).tolist()
                noise_y += (cord_y+dy).tolist()
            
            random.shuffle(noise_list)
            for cord_x, cord_y in noise_list[:int(len(noise_list) * (1-percent))]:
                dx = np.random.randint(-delta_d,delta_d)
                dy = np.random.randint(-delta_d,delta_d)
                noise_x += (cord_x+dx).tolist()
                noise_y += (cord_y+dy).tolist()
        
        noise_x, noise_y = np.clip(np.array(noise_x), 0, M-1), np.clip(np.array(noise_y), 0, N-1)
        noise_mat = coo_matrix((np.ones(noise_x.shape), (noise_y, noise_x)), shape=(self.hts.shape[0], self.freqs.shape[0])).toarray()
        
        if np.random.uniform(0,1) < 0.6:
            noise_mat = morphology.dilation(noise_mat>0, np.array([[0,1,0],[1,1,1],[1,1,1],[0,1,0]])) + 0
        return noise_mat

    def remove_nosise_around_blocks(self, noise_map, omode, xmode):
        if np.random.uniform(0,1) < 0.85:
            mask = 1- morphology.dilation((xmode + omode)>0, np.ones([np.random.randint(4,6),np.random.randint(4,6)]))
            noise_map *= mask

    def add_break_in_trace(self, fvs, vhs, mat, p = 0.5):
        # break_region = np.ones_like(mat)
        for _ in range(np.random.randint(0,10)):
            if np.random.uniform(0,1) > p:
                continue
            i = np.random.randint(0,fvs.shape[0]-1)
            ivh, ifv = np.argmin(np.abs(self.hts - vhs[i])), np.argmin(np.abs(self.freqs - fvs[i]))
            dx = np.random.randint(1,5)
            x1 = max(0,ifv-dx//2)
            x2 = min(ifv+dx//2, mat.shape[1])
            y1 = max(0,ivh-20//dx)
            y2 = min(ivh+20//dx, mat.shape[0])
            mat[y1:y2, x1:x2] = 0
        return mat

    def add_break_at_asym_freq(self, asym_f, fvs, vhs, mat, p = 0.2):
        # break_region = np.ones_like(mat)
        for fa in asym_f:
            if np.random.uniform(0,1) < p:
                i = np.argmin(np.abs(fvs - fa)) - 1
                ifa = np.argmin(np.abs(self.freqs - fa))
                iva = np.argmin(np.abs(self.hts - vhs[i]))
                mat[iva-5:iva+5, ifa-5:ifa+5] = 0
        return mat

    def graph_generate(self, fes, Z, dz = 0.5, savedata = False):
        fps,ppama=self.sampling_fps(Z, mode=np.random.choice([0, 1, 2, 3, 4], p=[0.22, 0.22, 0.12, 0.22, 0.22]))
        fvsO,vhsO,fvsX,vhsX=OXmode_cython(fps,fes,dz) # x1 y1 x2 y2 
        
        asym_fO = []
        asym_fX = []
        for i in range(1,fvsO.shape[0]):
            if vhsO[i] - vhsO[i-1] > 80:
                asym_fO += [fvsO[i]]
        for i in range(1,fvsX.shape[0]):
            if vhsX[i] - vhsX[i-1] > 80:
                asym_fX += [fvsX[i]]

        fo_peaks, fo_peaks_properties = find_peaks(vhsO,prominence=[3,None],width=1)
        fx_peaks, fx_peaks_properties = find_peaks(vhsX,prominence=[3,None],width=1)
        
        rand_val = np.random.uniform(0, 1)
        if rand_val < 0.3:  # 30% chance cutting at the last cusps
            if fo_peaks.shape[0] > 0 and fx_peaks.shape[0] > 0:
                fo_start = fvsO[fo_peaks[0]]
                fx_start = fvsX[fx_peaks[0]]
            else:
                fo_start = self.freqs[0]
                fx_start = self.freqs[0]
        elif rand_val < 0.5:  # 20% chance None (starts at beginning)
            fo_start = self.freqs[0]
            fx_start = self.freqs[0]
        else:  # 50% chance random cutoff between 0 and 100 to simulate low frequency absorption
            start_idx = np.random.randint(0, 100)
            fo_start = self.freqs[start_idx]
            fx_start = self.freqs[start_idx]

        O_trace = self.Coo2Mat(fvsO,vhsO, fs_start=fo_start)
        X_trace = self.Coo2Mat(fvsX,vhsX, fs_start=fx_start)

        ivhsO, ifvsO = np.nonzero(O_trace)
        ivhsX, ifvsX = np.nonzero(X_trace)

        # # Generate noisy data
        sigma_x = np.random.uniform(1.5,2.0)
        sigma_y = sigma_x * np.random.uniform(1.1,1.5) #np.random.uniform(1.8,2.2)
        fvsO_noisy_x, vhsO_noisy_y = self.scatter_around_points(np.array(ifvsO), np.array(ivhsO), sigma_x=sigma_x, sigma_y=sigma_y)
        fvsX_noisy_x, vhsX_noisy_y = self.scatter_around_points(np.array(ifvsX), np.array(ivhsX), sigma_x=sigma_x, sigma_y=sigma_y)

        # print(fvsO_noisy_x.shape[0] == vhsO_noisy_y.shape[0])        
        
        omode = coo_matrix((np.ones(fvsO_noisy_x.shape), (vhsO_noisy_y,fvsO_noisy_x)), 
                            shape=(self.hts.shape[0], self.freqs.shape[0])).toarray()
        xmode = coo_matrix((np.ones(fvsX_noisy_x.shape), (vhsX_noisy_y,fvsX_noisy_x)), 
                            shape=(self.hts.shape[0], self.freqs.shape[0])).toarray()

        # struct = np.array([[0,1,0],[1,1,1],[0,1,0]])
        struct = (np.random.rand(3, 3)<0.5).astype(np.int_)
        struct[1,1] = 1
        omode = morphology.closing(omode, struct)
        xmode = morphology.closing(xmode, struct)
        
        self.adding_holes_in_trace(omode)
        self.adding_holes_in_trace(xmode)
        
        noise_map = self.gen_noise_blocks()
        if np.random.uniform(0,1) < 0.5: 
            noise_map = self.add_noise_lines(noise_map)
            
        omode_clean = np.copy(omode)
        xmode_clean = np.copy(xmode)
        
        # 70% chance to add horizontal broadcast interference noise
        if np.random.uniform(0,1) < 0.7:
            noise_map = self.add_horizontal_noise_lines(noise_map)
        
        multihop_map = self.creating_multihop(fvsO, vhsO, fo_start)#fmin=foe+1) 
        multihop_map += self.creating_multihop(fvsX, vhsX, fx_start)# fmin=foe+1)

        self.remove_nosise_around_blocks(noise_map, omode, xmode)
        self.remove_nosise_around_blocks(multihop_map, omode, xmode)
        
        if np.random.uniform(0,1) < 0.4: #30% chance no multihop
            multihop_map = np.zeros(multihop_map.shape, dtype=np.uint8)
        else:
            multihop_map = multihop_map.astype(np.uint8)

        self.add_break_in_trace(fvsO,vhsO,omode)
        self.add_break_in_trace(fvsX,vhsX,xmode)

        self.add_break_at_asym_freq(asym_fO, fvsO,vhsO,omode)
        self.add_break_at_asym_freq(asym_fX, fvsX,vhsX,xmode)

        omode = morphology.remove_small_objects(omode>0, min_size=10, connectivity=1)
        xmode = morphology.remove_small_objects(xmode>0, min_size=10, connectivity=1)

        omode = morphology.remove_small_holes(omode, area_threshold=10, connectivity=1)
        xmode = morphology.remove_small_holes(xmode, area_threshold=10, connectivity=1)
        # Optional random rotation to make examples more general
        if np.random.rand() < 0.5:
            from scipy.ndimage import rotate
            angle = np.random.uniform(-5, 5)
            omode = rotate(omode, angle, reshape=False, order=0)
            xmode = rotate(xmode, angle, reshape=False, order=0)
            multihop_map = rotate(multihop_map, angle, reshape=False, order=0)
            omode_clean = rotate(omode_clean, angle, reshape=False, order=0)
            xmode_clean = rotate(xmode_clean, angle, reshape=False, order=0)

        if np.sum(np.sum(omode_clean, axis=0) > 0) > 2/3 * self.freqs.shape[0] and np.random.rand() < 0.10:
            crop_idx = np.random.randint(int(0.1 * self.freqs.shape[0]), int(0.4 * self.freqs.shape[0]))
            omode[:, :crop_idx] = 0
            xmode[:, :crop_idx] = 0
            omode_clean[:, :crop_idx] = 0
            xmode_clean[:, :crop_idx] = 0

        if np.random.uniform(0,1) < 0.05:# 0.05: # 5% chance negative data
            omode = np.zeros(omode.shape, dtype=np.uint8)
            xmode = np.zeros(xmode.shape, dtype=np.uint8)
            omode_clean = np.zeros(omode_clean.shape, dtype=np.uint8)
            xmode_clean = np.zeros(xmode_clean.shape, dtype=np.uint8)
            multihop_map = np.zeros(multihop_map.shape, dtype=np.uint8)

        fig,axes = plt.subplots(1,4,figsize=(23.0,6.0),sharex=False)
        ax1 = axes[0]
        ax1.plot(fps,Z,label='Plasma frequency')
        ax1.plot(fvsO,vhsO,label='O-mode')
        ax1.plot(fvsX,vhsX,label='X-mode')
        ax1.set_ylim(0,self.hts.max())
        ax1.set_title(r'$f_p$ profile and O- X- traces')
        ax1.set_xlabel('Frequency [MHz]')
        ax1.set_ylabel('Height [km]')
        ax1.legend()
        if fo_peaks.shape[0] > 0:
            ax1.plot(fvsO[fo_peaks],vhsO[fo_peaks],'x')
        if fx_peaks.shape[0] > 0:
            ax1.plot(fvsX[fx_peaks],vhsX[fx_peaks],'x')

        ax2 = axes[1]
        
        # Plot O-mode Segmentation Mask
        ax2.imshow(omode_clean > 0, cmap='viridis', origin='lower', interpolation='nearest')
        ax2.set_title('O-mode Mask')
        ax2.set_xlim(0, self.freqs.shape[0])
        ax2.set_ylim(0, self.hts.shape[0])

        ax3 = axes[2]
        # Plot X-mode Segmentation Mask
        ax3.imshow(xmode_clean > 0, cmap='viridis', origin='lower', interpolation='nearest')
        ax3.set_title('X-mode Mask')
        ax3.set_xlim(0, self.freqs.shape[0])
        ax3.set_ylim(0, self.hts.shape[0])

        ax4 = axes[3]
        ax4.imshow((omode+xmode+noise_map+multihop_map)>0,origin='lower', interpolation='nearest')
        ax4.set_title('Synthetic ionogram')
        ax4.set_axis_off()

        if savedata == True: 
            file_name = "_".join([f"{x:.5f}" for x in ppama]) + "_.png"
            fig.savefig(f"{self.training_dir}/visual/{file_name}", dpi=100, bbox_inches='tight', pad_inches=0, facecolor=fig.get_facecolor())
            plt.close()
            
            imgTrain = Image.fromarray(np.flip((omode+xmode+noise_map+multihop_map)>0,axis=0).astype(dtype=np.uint8)*255, 'L')
            imgTrain.save(f"{self.training_dir}/train/{file_name}")
            
            np.save(f"{self.training_dir}/npy/{file_name}", {
                'omode_mask': (omode_clean>0).astype(np.uint8),
                'xmode_mask': (xmode_clean>0).astype(np.uint8)
            })
            # omode_data = np.zeros((*omode.shape,3), dtype=np.uint8)
            # omode_data[:,:,0] = np.flip(O_trace>0,axis=0) * 255
        
            # xmode_data = np.zeros((*xmode.shape,3), dtype=np.uint8)
            # xmode_data[:,:,1] = np.flip(X_trace>0,axis=0) * 255

            # imgO = Image.fromarray(omode_data, 'RGB')
            # imgX = Image.fromarray(xmode_data, 'RGB')

            # imgO.save(f"{self.training_dir}/O/{file_name}")
            # imgX.save(f"{self.training_dir}/X/{file_name}")
        else:
            plt.show()
        return 
        # np.save(f"{self.training_dir}/asym_f/{file_name}", {'asym_fO':asym_fO, 'asym_fX':asym_fX})

if __name__ == "__main__":
    import datetime
    from ISRpy import beampack
    import scipy.constants as const

    jro0 = beampack.jrobeam
    training_dir = 'training_data'

    d = np.load('JM91J_2016014141141.npy', allow_pickle=True)[()]
    freqs = d['vipir_freqs']
    hts = d['vipir_hts']

    noise_dir = 'VIPIR_PROC/Dog_noise/'
    noise_files = glob.glob1(noise_dir,'*.npy')
    # print(noise_files)
    ipp = 5.6e3*0.15
    print(ipp)
    # ipp - hts[-1]*1
    gt = GenTrain(freqs,hts,ipp,training_dir,noise_dir)
    dt0 = datetime.datetime(2016,1,14)
    igrf_year = dt0.year + (dt0-datetime.datetime(dt0.year,1,1)).total_seconds() / (
            datetime.datetime(dt0.year+1,1,1)-datetime.datetime(dt0.year,1,1)).total_seconds()

    dz=0.5 # resolution in km --- 0.1 is for good resolution, 0.5 for speed
    zmax = 800
    Z = np.arange(0.01,zmax,dz)

    fes = []
    for iZ in Z:
        r,geoc_lat,lon,aspect,B = jro0.aspect_angle(igrf_year,jro0.xyz0 + jro0.zenith0 * iZ)
        fes += [const.e * np.linalg.norm(B) * 1e-9 / const.m_e /1e6 /2/np.pi] #MHz
    fes = np.array(fes)

    gt.graph_generate(fes, Z, dz, savedata = False)

    if True:
        from joblib import Parallel, delayed
        import multiprocessing as mp

        para_func = lambda x : gt.graph_generate(fes, Z, dz, savedata = True)

        # Use all available CPU cores to generate the graphs in parallel
        Parallel(n_jobs=mp.cpu_count()-0)(delayed(para_func)(_) for _ in range(40000))

