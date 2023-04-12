import cv2 as cv
import os


def rescaleFrame(frame, scale=.05):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimensions = (width, height)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)


for filename in os.listdir('OpenCV Course/Photos'):
    ext = os.path.splitext(filename)[-1].lower()

    if ext == ".jpg":
        filepath = os.path.join('OpenCV Course/Photos/', filename)
        img = cv.imread(filepath, cv.IMREAD_UNCHANGED)
        img_resized = rescaleFrame(img)
        image = cv.rotate(img_resized, cv.ROTATE_90_CLOCKWISE)

        cv.imshow("UNO", image)
        cv.waitKey(0)
        cv.destroyAllWindows()
