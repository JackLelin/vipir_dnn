import glob, copy
import matplotlib
import numpy as np
from matplotlib.backend_bases import MouseButton
import matplotlib.pyplot as plt
#import tkinter
#import matplotlib
#matplotlib.use('TkAgg')
from matplotlib.widgets import Slider, Button, RadioButtons

from sklearn.cluster import AgglomerativeClustering
from skimage import measure, morphology

from scipy import ndimage

outdir = 'E:/Machine_Learning/Parabola_Train/TOPHAT/handpick/'
genedfiles = glob.glob1(outdir,'*.npy') 

orgfolder = 'E:/Machine_Learning/Parabola_Train/TOPHAT/npy/'
selected_files = glob.glob1('E:/Machine_Learning/Parabola_Train/TOPHAT/selected/','*.png') 
selected_files.sort()

todofiles = []
for file in selected_files:
    npyfile = file[:-4] + '.npy'
    if npyfile not in genedfiles:
        todofiles += [npyfile]

print(len(todofiles), len(selected_files))

file_idx = 0

def loadnpyfile(val):
    global Xcord, Ycord, Label, Label_old, original
    global XcordO, YcordO, LabelO, LabelO_old
    
    cleaned = np.load(orgfolder+todofiles[file_idx],allow_pickle=True)[()]
    original = cleaned['original']
    # originalO = cleaned['originalO']
    
    Xcord, Ycord, Label = getLabel(original)
    # XcordO, YcordO, LabelO = getLabel(originalO)

    Label_old = np.copy(Label)
    # LabelO_old = np.copy(LabelO)

def getLabel(original):
    crop_region = np.zeros_like(original)
    crop_region[120:-10,30:390] = 1
    original = original * crop_region
    footprint = np.array([5,3])
    w_tophat = ndimage.white_tophat(original, footprint)
    w_tophat[w_tophat<0] = 0
    
    thresh = w_tophat > np.percentile(w_tophat,94,axis=0)

    temp3 = morphology.remove_small_objects(thresh, min_size=5, connectivity=1)

    ycord, xcord = np.nonzero(temp3)
    
    X_points = np.vstack([xcord,ycord]).T
    
    
    ###########################################################
    from sklearn.cluster import AgglomerativeClustering
    from skimage import measure

    label_img, num = measure.label(temp3, return_num=True)

    clustering = AgglomerativeClustering(linkage='single', n_clusters= int(num * 0.75))
    clustering.fit(X_points)
    
    cluster_labels = clustering.labels_
    cluster_labels += 1
    
    unique_labels = np.unique(cluster_labels)
    cluster_size = np.array([np.sum( cluster_labels == label) for label in unique_labels])
        
    return xcord, ycord, cluster_labels

def drawplot():
    global im
    im = ax.scatter(Xcord, Ycord, c= Label,cmap=cmap)
    im.set_clim([0,Label.max()])
    im.axes.set_title(todofiles[file_idx]); im.axes.title.set_fontsize(10.0)

#cmap = copy.copy(plt.cm.get_cmap('jet'))
cmap = copy.copy(matplotlib.colormaps.get_cmap('jet'))
cmap.set_over([1.0,0,0,1.0])
cmap.set_under([0,0,0,1.0])

loadnpyfile(0)

# print(type(mask_full))
fig=plt.figure(figsize=(10,6))      
plt.subplots_adjust(left=0.15)
ax = fig.subplots()
drawplot()
# plt.colorbar()
axcolor = 'lightgoldenrodyellow'

rax = plt.axes([0.025, 0.5, 0.08, 0.16], facecolor=axcolor)
radio = RadioButtons(rax, ('CC', 'large','median','range'), active=0)

nextbuttax = plt.axes([0.8, 0.025, 0.1, 0.04])
nextbutton = Button(nextbuttax, 'Next', color=axcolor, hovercolor='0.975')

def loadnext(event):
    global file_idx
    file_idx+=1

    ax.cla()
    loadnpyfile(0)
    drawplot()

def mouselabel(cords, action):  
    global Label, Xcord, Ycord

    X_points = np.vstack([Xcord,Ycord]).T
    idx = np.argmin(np.sum((X_points-cords)**2, axis=1))

    if action == 1:
        Label[Label_old == Label_old[idx]] = -1
    if action == 0:
        Label[Label_old == Label_old[idx]] = Label_old[idx]

    im.set_array(Label)
    fig.canvas.draw_idle()

def mousepattern(cords, action):  # change color of selected area
    global Label, Xcord, Ycord

    X_points = np.vstack([Xcord,Ycord]).T
    dist = np.sum((X_points-cords)**2, axis=1)**0.5

    indices = np.where(dist<9)[0]
    

    if action == 1:
        Label[indices] = -1
    if action == 0:
        Label[indices] = Label_old[indices]

    im.set_array(Label)
    fig.canvas.draw_idle()
    
def mousemedian(cords, action):  # change color of selected area
    global Label, Xcord, Ycord

    X_points = np.vstack([Xcord,Ycord]).T
    dist = np.sum((X_points-cords)**2, axis=1)**0.5

    indices = np.where(dist<4)[0]
    

    if action == 1:
        Label[indices] = -1
    if action == 0:
        Label[indices] = Label_old[indices]

    im.set_array(Label)
    fig.canvas.draw_idle()

def mousebrush(cords, action):  # change color of selected area
    global Label, Xcord, Ycord

    X_points = np.vstack([Xcord,Ycord]).T
    dist = np.sum((X_points-cords)**2, axis=1)**0.5

    indices = np.where(dist<1.1)[0]

    if action == 1:
        Label[indices] = -1
    if action == 0:
        Label[indices] = Label_old[indices]

    im.set_array(Label)
    fig.canvas.draw_idle()


nextbutton.on_clicked(loadnext)


def on_click(event):
    action = None
    if event.button is MouseButton.LEFT:
        action = 1
    if event.button is MouseButton.RIGHT:
        action = 0

    if event.inaxes == ax:
        if radio.value_selected == 'CC':
            mouselabel(np.array([[event.xdata, event.ydata]]), action)
        if radio.value_selected == 'large':
            mousepattern(np.array([[event.xdata, event.ydata]]), action)
        if radio.value_selected == 'median':
            mousemedian(np.array([[event.xdata, event.ydata]]), action)
        if radio.value_selected == 'range':
            mousebrush(np.array([[event.xdata, event.ydata]]), action)
        
plt.connect('button_press_event', on_click)

savebutax = plt.axes([0.15, 0.025, 0.1, 0.04])
savebutton = Button(savebutax, 'Save', color=axcolor, hovercolor='0.975')

def saveresult(event):
    global original, Xcord, Ycord, Label

    mask_full = np.zeros_like(original,dtype=np.int32)
    mask_full[(Ycord[Label == -1],Xcord[Label == -1])] = 1

    fig=plt.figure(figsize=(10,13))      

    plt.subplot(2,1,1)
    im = plt.imshow(np.log10(np.abs(original)),cmap='binary',origin='lower',aspect='auto')
    im.axes.set_title(todofiles[file_idx]); im.axes.title.set_fontsize(10.0)
    plt.colorbar()

    plt.subplot(2,1,2)
    im = plt.imshow(mask_full,cmap='binary',origin='lower',aspect='auto')
    plt.colorbar()
    
    print('saving figure', todofiles[file_idx])    
    fig.savefig(outdir+todofiles[file_idx][:-4],bbox_inches='tight')
    plt.show()

    saved = {'mask_full':mask_full}
    np.save(outdir+todofiles[file_idx][:-4],saved)

savebutton.on_clicked(saveresult)

plt.show()