import numpy as np

topLeft = (30,50)
topRight = (400,40)
botRight = (410,500)
botLeft = (40,490)

pts = np.array([topLeft,topRight,botRight,botLeft])

rect = np.zeros((4,2), dtype= "float32")
s = pts.sum(axis=1)
print(s)
rect[0] = pts[np.argmin(s)]
print(rect)
rect[2] = pts[np.argmax(s)]
print(rect)
diff = np.diff(pts, axis=1)
print(diff)
rect[1] = pts[np.argmin(diff)]
print(rect)
rect[3] = pts[np.argmax(diff)]
print(rect)