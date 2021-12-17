import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from skimage.exposure import match_histograms
from cv2.ximgproc import guidedFilter

path = './input'
f_names = ['lr_red.tif', 'lr_green.tif', 'lr_blue.tif', 'lr_nir.tif', 'pan.tif']
img_set = []
for f in f_names:
    img = plt.imread(os.path.join(path, f))
    img_set.append(img)

pan_h = img_set[-1].shape[0]
pan_w = img_set[-1].shape[1]
ms_img = np.zeros((pan_h, pan_w, 4), dtype=np.uint16)

for i in range(5):
    # img_set[i] = (img_set[i] - np.min(img_set[i])) / (np.max(img_set[i]) - np.min(img_set[i]))
    if i != 4:
        img_set[i] = np.uint16(cv2.resize(img_set[i], dsize=img_set[-1].shape, interpolation=cv2.INTER_CUBIC))
        ms_img[..., i] = img_set[i]

p = PCA(4)
pca_ms = p.fit_transform(np.reshape(ms_img, (pan_h * pan_w, 4)))
print(pca_ms.shape)

pca_1 = pca_ms[:, 0]
pan = img_set[-1]

# pan = guidedFilter(np.reshape(pca_1, (pan_h, pan_w)), pan, 4, 0.1)
pan = match_histograms(pan, np.reshape(pca_1, (pan_h, pan_w)))


pca_ms[:, 0] = np.reshape(pan, (pan_h * pan_w))
pca_ms = p.inverse_transform(pca_ms)
pca_ms = np.reshape(pca_ms, (pan_h, pan_w, 4))
print(pca_ms.dtype)
#
plt.imshow(pca_ms[..., 0])
plt.show()
print(np.min(pca_ms.astype(np.uint16)), np.max(pca_ms.astype(np.uint16)))
#
out_name = ['red.tif', 'green.tif', 'blue.tif', 'nir.tif']
for i in range(4):
    im = cv2.imwrite(os.path.join('output_sharpen', out_name[i]), pca_ms[..., i].astype(np.uint16))




