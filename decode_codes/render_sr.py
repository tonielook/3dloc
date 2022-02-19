# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 22:55:18 2022

@author: DAILingjia
"""


""" 
Explanation: 6_render_v2.py
Loading function: render_v2.py
"""
import torch
import pandas as pd
import numpy as np
from PIL import Image
np.set_printoptions(suppress=True)
torch.set_printoptions(sci_mode=False)

from render_v2 import Renderer2D_v2


# Sub HR Images with Part of All frames
nt = 9000


# Ground-truth Figure4a - DECODE
# ------------------------------------------------
csv_path = './DECODE_ROI3_LD_DC.csv'

em_decode = pd.read_csv(csv_path)
em_decode = torch.tensor(em_decode.values)

# em_decode[0]
keys = ['emitter_idx','frame_idx','x_nm','y_nm','z_nm','intensity','x_sig','y_sig','z_sig']
for idx in range(em_decode.shape[1]):
    print(f'{idx} {keys[idx]}: min = {torch.min(em_decode[:,idx])}, max = {torch.max(em_decode[:,idx])}')
""" Info
0 emitter_idx: min = 93.0,  max = 7766833.0
1 frame_idx:   min = 1.0,   max = 149999.0
2 x_nm:        min = 0.007, max = 10529.993
3 y_nm:        min = 0.013, max = 11429.996
4 z_nm:        min = -393,  max = 400
5 intensity:   min = 44273, max = 998495
6 x_sig:       min = 0.012, max = 0.493
7 y_sig:       min = 0.012, max = 0.326
8 z_sig:       min = 0.011, max = 0.127
"""

render_v2 = Renderer2D_v2(colextent=[-500,500],px_size=10.,sigma_blur=0.,
                          rel_clip=0.05,contrast=2)

_,a = render_v2.render(em_decode[:,2:5], em_decode[:,4])

em_decode_sub = em_decode[em_decode[:,1]<nt]
print(len(em_decode_sub))
# nt=9k, 94885
_,a = render_v2.render(em_decode_sub[:,2:5], em_decode_sub[:,4])

img = Image.fromarray(np.uint8(a*255), 'RGB')
# Modify save path here!
save_name = r'C:\Users\DLJ\Desktop\hr\gt_9k.png'
img.save(save_name)





# Prediction Figure4
# --------------------------------------------------
# path from post-process
csv_path = 'pred_label.csv'

est = pd.read_csv(csv_path)
est = torch.tensor(est.values)

# est[0]
keys_pred = ['frame_idx','x_px','y_px','z_px','flux','xs','ys','zs','xer','yer','zer']
for idx in range(est.shape[1]):
    print(f'{idx} {keys_pred[idx]}: min = {torch.min(est[:,idx])}, max = {torch.max(est[:,idx])}')
""" Info - xy in pixel, z in unit of zeta
0 frame_idx: min = 1.0,    max = 149999.0
1 x_px:      min = -2.0,   max = 84.0
2 y_px:      min = -6.0,   max = 100.0
3 z_px:      min = 1.0,    max = 12.0
4 flux:      min = 6.5405, max = 454.5089
5 xs:        min = -0.5,   max = 0.5
6 ys:        min = -0.5,   max = 0.5
7 zs:        min = -0.5,   max = 0.5
"""

# change pixel locations to nm case
zmax = 6
zeta_unit = 800/10.3958
px_size = [127,117]

est_tmp = est.clone().detach()

# choice 1 - Without Floating Shift
est_tmp[:,1] = est_tmp[:,1] * px_size[0]
est_tmp[:,2] = est_tmp[:,2] * px_size[1]
est_tmp[:,3] = (est_tmp[:,3] - zmax) * zeta_unit

# choice 2 -  With Floating Shift
est_tmp[:,1] = (est_tmp[:,1] + est_tmp[:,5]) * px_size[0]
est_tmp[:,2] = (est_tmp[:,2] + est_tmp[:,6]) * px_size[1]
est_tmp[:,3] = (est_tmp[:,3] + est_tmp[:,7] - zmax) * zeta_unit

# est_tmp[0]
for idx in range(est_tmp.shape[1]):
    print(f'{idx} {keys_pred[idx]}: min = {torch.min(est_tmp[:,idx])}, max = {torch.max(est_tmp[:,idx])}')
"""Info
0 frame_idx:  min = 1.0,       max = 149999.0
1 x_px:       min = -192.8495, max = 10668.0
2 y_px:       min = -702.0,    max = 11700.0
3 z_px:       min = -384.771,  max = 461.725
4 flux:       min = 6.5405,    max = 454.5089
"""

render_v2 = Renderer2D_v2(colextent=[-500,500],px_size=10.,sigma_blur=0., 
                          rel_clip=0.005, contrast=2, xextent=(0,10600), 
                          yextent=(0,11450), zextent=(-400,400))

_,b = render_v2.render(est_tmp[:,1:4], est_tmp[:,3])

est_sub = est_tmp[est_tmp[:,0]<nt]
print(len(est_sub))
_,b = render_v2.render(est_sub[:,1:4], est_sub[:,3])

img = Image.fromarray(np.uint8(b*255), 'RGB')
save_name = r'C:\Users\DLJ\Desktop\hr\est_9k.png'
img.save(save_name)





# Prediction Figure4 UPSAMPLE
# ---------------------------------------------------
csv_path = r'D:\OneDrive - City University of Hong Kong\3dLoc\test_result\0104_DECODE_figure4a\data\pred_label_up_20k.csv'
# Parameters for Upsample
up_xy = 2
up_z = 250


csv_path = r'C:\Users\DLJ\Desktop\0114_figure4a_v4_upsample\pred_label_bol.csv'
up_xy = 4
up_z = 250

est = pd.read_csv(csv_path)
est = torch.tensor(est.values)

# est[0]
keys_pred = ['frame_idx','x_px','y_px','z_px','flux','xs','ys','zs','xer','yer','zer']
for idx in range(est.shape[1]):
    print(f'{idx} {keys_pred[idx]}: min = {torch.min(est[:,idx])}, max = {torch.max(est[:,idx])}')
""" Info
0 frame_idx: min = 1.0,       max = 20001.0
1 x_px:      min = 28.0,      max = 196.0
2 y_px:      min = 23.0,      max = 226.0
3 z_px:      min = 1.0,       max = 232.0
4 flux:      min = 7.6679,    max = 383.5652
5 xs:        min = -0.5,      max = 0.5
6 ys:        min = -0.5,      max = 0.5
7 zs:        min = -0.5,      max = 0.5
8 xer:       min = -130.5545, max = 678.0
9 yer:       min = -137.0933, max = 678.0
10 zer:      min = -85.8089,  max = 678.0
"""


# px -> nm
zmax = 6
zeta_unit = 800/10.3958
px_size = [127,117]

est_tmp = est.clone().detach()
est_tmp[:,1] = ((est_tmp[:,1] + est_tmp[:,5])/up_xy-14) * px_size[0]
est_tmp[:,2] = ((est_tmp[:,2] + est_tmp[:,6])/up_xy-14) * px_size[1]
est_tmp[:,3] = ((est_tmp[:,3] + est_tmp[:,7])/up_z*13 - zmax) * zeta_unit

# est_tmp[0]
for idx in range(est_tmp.shape[1]):
    print(f'{idx} {keys_pred[idx]}: min = {torch.min(est_tmp[:,idx])}, max = {torch.max(est_tmp[:,idx])}')
""" Info
0 frame_idx: min = 1.0, max = 20001.0
1 x_px:      min = 0.0, max = 10668.0
2 y_px:      min = -309.57, max = 11606.07
3 z_px:      min = -456.345, max = 468.565
4 flux:      min = 7.668, max = 383.565
5 xs:        min = -0.5, max = 0.5
6 ys:        min = -0.5, max = 0.5
7 zs:        min = -0.5, max = 0.5
8 xer:       min = -130.555, max = 678.0
9 yer:       min = -137.093, max = 678.0
10 zer:      min = -85.8089, max = 678.0
"""

render_v2 = Renderer2D_v2(colextent=[-500,500], px_size=10., sigma_blur=0., 
                          rel_clip=0.05, contrast=2,xextent=(0,10600), 
                          yextent=(0,11450), zextent=(-400,400))

est_sub = est_tmp[est_tmp[:,0]<nt]
print(len(est_sub))
_,b = render_v2.render(est_sub[:,1:4], est_sub[:,3])

img = Image.fromarray(np.uint8(b*255), 'RGB')
save_name = r'C:\Users\DLJ\Desktop\hr\est_9k_up4.png'
img.save(save_name)
