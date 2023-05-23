import numpy as np
width = np.sqrt((tl[0]-tr[0]) **2 + (tl[1]-tr[1]) **2)
hight = np.sqrt((tr[0]-br[0]) **2 + (tr[1]-br[1]) **2)

tl = (0,0)

pts = []

for c in corners: 
    euc = []
    for cnt in contours: 
        euc.append(np.sqrt((c[0]-cnt[0]) **2 + (c[1]-cnt[1]) **2))
    pts.append(euc[np.argmin(euc)])

for cnt in contours: 
    euc = np.sqrt((corners[0][0]-cnt[0]) **2 + (tl[1]-tr[1]) **2)


index_sort = sorted(range(len(contours)), key=lambda i : cv.contourArea(contours[i]), reverse=True)
cnt_sorted = []
for i in index_sort:
    cnt_sorted.append(contours[i])