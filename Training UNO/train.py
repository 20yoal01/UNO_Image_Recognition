import cv2 as cv
import numpy as np
import os
import random
import math


UNO_CARDS_PATH = 'OpenCV Course/Photos V2/'
OUTPUT = 'UNO Syn/images'
LABEL_PATH = 'UNO Syn/labels'
BACKGROUND_PATH = 'OpenCV Course/background'
CARD_TYPE = ['RED', 'GREEN', 'BLUE', 'YELLOW', 'WILD']

# IMAGES_PER_CARD = 200
# TOTAL_CARDS_TO_GENERATE = 56


IMAGES_PER_CARD = 200
TOTAL_CARDS_TO_GENERATE = 56

# DEFAULT
# (720, 1280)
# [-180, 180]
# [(TARGET_WIDTH//2) * 0.06, (TARGET_HEIGHT//2) * 0.06]
# (-0.35, 0.35)
# (-0.35, 0.35)
# (0, 100)
# (-30, 255)


TARGET_HEIGHT, TARGET_WIDTH = (720, 1280)
ROTATE_RANGE = [-180, 180]
TRANSLATE_RANGE = [(TARGET_WIDTH//2) * 0.10, (TARGET_HEIGHT//2) * 0.10]
PROJECTION_RANGE = (-0.60, 0.60)
SATURATION_RANGE = (0, 100)
VALUE_RANGE = (-30, 255)

class_name_to_id_mapping_symbol = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "d2": 10,
    "r": 11,
    "s": 12,
    "d4": 13,
    "wild_card": 14,
    "wild_custom": 15,
    "wild_shuffle": 16
}

class_name_to_id_mapping_card_set = {
    "BLUE zero": 0,
    "BLUE one": 1,
    "BLUE two": 2,
    "BLUE three": 3,
    "BLUE four": 4,
    "BLUE five": 5,
    "BLUE six": 6,
    "BLUE seven": 7,
    "BLUE eight": 8,
    "BLUE nine": 9,
    "BLUE d2": 10,
    "BLUE r": 11,
    "BLUE s": 12,
    "RED zero": 13,
    "RED one": 14,
    "RED two": 15,
    "RED three": 16,
    "RED four": 17,
    "RED five": 18,
    "RED six": 19,
    "RED seven": 20,
    "RED eight": 21,
    "RED nine": 22,
    "RED d2": 23,
    "RED r": 24,
    "RED s": 25,
    "YELLOW zero": 26,
    "YELLOW one": 27,
    "YELLOW two": 28,
    "YELLOW three": 29,
    "YELLOW four": 30,
    "YELLOW five": 31,
    "YELLOW six": 32,
    "YELLOW seven": 33,
    "YELLOW eight": 34,
    "YELLOW nine": 35,
    "YELLOW d2": 36,
    "YELLOW r": 37,
    "YELLOW s": 38,
    "GREEN zero": 39,
    "GREEN one": 40,
    "GREEN two": 41,
    "GREEN three": 42,
    "GREEN four": 43,
    "GREEN five": 44,
    "GREEN six": 45,
    "GREEN seven": 46,
    "GREEN eight": 47,
    "GREEN nine": 48,
    "GREEN d2": 49,
    "GREEN r": 50,
    "GREEN s": 51,
    "d4": 52,
    "wild_card": 53,
    "wild_custom": 54,
    "wild_shuffle": 55
}


DATASPLIT = {
    'Train': 0.7,
    'Validation': 0.15,
    'Test': 0.15
}


def resize(img, scale_percent=.05):
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)


def resize_no_aspect(img, width, height):
    dim = (width, height)
    return cv.resize(img, dim, interpolation=cv.INTER_AREA)

# def rotate(img, angle, rotPoint=None):
#     (height, width) = img.shape[:2]

#     if rotPoint is None:
#         rotPoint = (width//2, height//2)

#     rotMat = cv.getRotationMatrix2D(rotPoint, angle, 0.8)
#     dimensions = (width, height)

#     return cv.warpAffine(img, rotMat, dimensions)

# def translate(img, x, y):
#     transMat = np.float32([[1, 0, x], [0, 1, y]])
#     dimensions = (img.shape[1], img.shape[0])
#     return cv.warpAffine(img, transMat, dimensions)


def projectiveTransform(img, angle, translation=(0, 0), projective=(0, 0), rotPoint=None):
    (height, width) = img.shape[:2]
    dimensions = (width, height)
    if rotPoint is None:
        rotPoint = (width // 2, height // 2)

    # Roterar bilden
    rotMat = cv.getRotationMatrix2D(rotPoint, angle, 1.0)
    rotMat = np.vstack([rotMat, [0, 0, 1]])

    # Translation
    transMat = np.float32(
        [[1, 0, translation[0]], [0, 1, translation[1]], [0, 0, 1]])

    # Beräknar våran nya perspektiv matris
    src_pts = np.float32([[0, 0], [width, 0], [width, height], [0, height]])
    dst_pts = np.float32([[0, 0], [width, 0], [
                         width + projective[0] * height, height], [-projective[1] * width, height]])
    perspMat = cv.getPerspectiveTransform(src_pts, dst_pts)

    # Kombinerar allting
    projMat = np.dot(transMat,  np.dot(perspMat, rotMat))

    return cv.warpPerspective(img, projMat, dimensions)


def changeHSV(img, saturation, value):
    changeImg = img.copy()
    changeImg = cv.cvtColor(changeImg, cv.COLOR_BGR2HSV).astype("float32")
    h, s, v = cv.split(changeImg)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    lim = 255 - saturation
    s[s > lim] = 255
    s[s <= lim] += saturation
    changeImg = cv.merge([h, s, v])
    changeImg = cv.cvtColor(changeImg.astype("uint8"), cv.COLOR_HSV2BGR)
    return changeImg


def extract_uno(img):
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    blur = cv.GaussianBlur(grey, (5, 5), cv.BORDER_DEFAULT)

    t_lower = 0  # Lower Threshold
    t_upper = 255  # Upper threshold
    aperture_size = 3  # Aperture size

    canny = cv.Canny(blur, t_lower, t_upper, apertureSize=aperture_size)

    # cv.imshow('canny', canny)

    # Dilate the edges to close any gaps in the white outline
    kernel = np.ones((5, 5), np.uint8)
    canny = cv.dilate(canny, kernel, iterations=1)

    contours, hierarchies = cv.findContours(
        canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    # cv.imshow('Canny', canny)

    mask = np.zeros(shape=img.shape, dtype='uint8')

    cv.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv.FILLED)
    x, y, w, h = cv.boundingRect(contours[0])

    blank = np.zeros(shape=img.shape, dtype='uint8')
    # blank[:] = (255, 255, 255)
    # cv.imshow('sh', mask)
    # mask_cropped = mask[y:y+h, x:x+w]
    # masked_img = img[y:y+h, x:x+w]
    cv.copyTo(img, mask, blank)
    # cv.imshow('m', blank)
    # cv.rectangle(blank, (135, 145), (blank.shape[1]//5,
    # blank.shape[0]//7), (0, 255, 0), thickness=3)
    # cv.normalize(mask.copy(), mask, 0, 255, cv.NORM_MINMAX)

    # cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
    # cv.imshow('test', img)
    return mask, (x, y, w, h)


def add_obj(background, img, mask, x, y):
    '''
    Argument: 
    background - bakgrunden som ska användas
    img - bilden på UNO-kortet "orginalet" 
    mask - masken som tog fram av tidigare metod 
    x,y - koordinaterna för mitten på bildobjektet. Dessa måste vara mindre än bakgrundens dimentioner
    '''

    bg = background.copy()

    h_bg, w_bg = bg.shape[0], bg.shape[1]
    h, w = img.shape[0], img.shape[1]

    x = x - int(w/2)
    y = y - int(h/2)

    mask_boolean = mask[:, :, 0] != 0
    mask_rgb_boolean = np.stack(
        [mask_boolean, mask_boolean, mask_boolean], axis=2)

    if x >= 0 and y >= 0:

        # h_part - part of the image which overlaps background along y-axis
        h_part = h - max(0, y+h-h_bg)
        # w_part - part of the image which overlaps background along x-axis
        w_part = w - max(0, x+w-w_bg)

        bg[y:y+h_part, x:x+w_part, :] = bg[y:y+h_part, x:x+w_part, :] * \
            ~mask_rgb_boolean[0:h_part, 0:w_part, :] + \
            (img * mask_rgb_boolean)[0:h_part, 0:w_part, :]

    return bg


index = 0
folder_index = 0
CURRENT_UNO_PATH = UNO_CARDS_PATH + CARD_TYPE[index]
total_cards = 0

train = DATASPLIT['Train']
validation = DATASPLIT['Validation']
test = DATASPLIT['Test']

IMAGES_TO_TRAIN = math.ceil(IMAGES_PER_CARD * train)
IMAGES_TO_VALIDATE = math.floor(IMAGES_PER_CARD * validation)
IMAGES_TO_TEST = math.floor(IMAGES_PER_CARD * test)

if not os.path.isdir(LABEL_PATH):
    os.makedirs(LABEL_PATH)

for datasplit in DATASPLIT:
    # DATASPLIT[datasplit]
    output_path = os.path.join(OUTPUT, datasplit)
    label_path = os.path.join(LABEL_PATH, datasplit)
    if not os.path.isdir(label_path):
        os.makedirs(label_path)
    if not os.path.isdir(output_path):
        print(output_path)
        os.makedirs(output_path)


def convert_to_yolov5(info_card, cardname, output_name, current_label_path):
    print_buffer = []

    try:
        class_id = class_name_to_id_mapping[cardname]
    except KeyError:
        print("Invalid Class. Must be one from ",
              class_name_to_id_mapping.keys())

    x, y, w, h = info_card

    # Transform the bbox co-ordinates as per the format required by YOLO v5
    x_max = x + w
    y_max = y + h

    b_center_x = (x + x_max) / 2
    b_center_y = (y + y_max) / 2
    b_width = w
    b_height = h

    # Normalise the co-ordinates by the dimensions of the image
    b_center_x /= TARGET_WIDTH
    b_center_y /= TARGET_HEIGHT
    b_width /= TARGET_WIDTH
    b_height /= TARGET_HEIGHT

    # Write the bbox details to the file
    print_buffer.append("{} {:.3f} {:.3f} {:.3f} {:.3f}".format(
        class_id, b_center_x, b_center_y, b_width, b_height))

    # Name of the file which we have to save
    save_file_name = os.path.join(current_label_path, output_name)

    # Save the annotation to disk
    print("\n".join(print_buffer), file=open(save_file_name, "w"))


while True:
    if total_cards >= TOTAL_CARDS_TO_GENERATE:
        break

    total_cards += 1

    filedir = os.listdir(CURRENT_UNO_PATH)

    if folder_index >= len(filedir) and index < len(CARD_TYPE):
        index += 1
        if index >= len(CARD_TYPE):
            break
        CURRENT_UNO_PATH = UNO_CARDS_PATH + CARD_TYPE[index]
        filedir = os.listdir(CURRENT_UNO_PATH)
        folder_index = 0

    filename = filedir[folder_index]

    cardType = CARD_TYPE[index] + " " + filename[:]
    print(cardType)

    folder_index += 1

    img_to_create = 0

    DATASET = ['Train', 'Validation', 'Test']
    dataset_index = -1
    dataset_size = 0
    dataset_current = 0

    filepath = os.path.join(CURRENT_UNO_PATH, filename)
    img = cv.imread(filepath)
    img_resized = resize(img, scale_percent=0.025)

    mask, card_info = extract_uno(img_resized)

    for background in os.listdir(BACKGROUND_PATH):
        if dataset_current >= dataset_size:
            dataset_current = 0
            dataset_index += 1

            if dataset_index > len(DATASET) - 1:
                break

            match DATASET[dataset_index]:
                case 'Train':
                    dataset_size = IMAGES_TO_TRAIN
                case 'Validation':
                    dataset_size = IMAGES_TO_VALIDATE
                case 'Test':
                    dataset_size = IMAGES_TO_TEST

        dataset_current += 1
        current_output = os.path.join(OUTPUT, DATASET[dataset_index] + '/')
        current_label_path = os.path.join(
            LABEL_PATH, DATASET[dataset_index] + '/')

        if img_to_create >= IMAGES_PER_CARD:
            break

        img_to_create += 1

        ext = os.path.splitext(filename)[-1].lower()
        if ext == ".jpg":
            background_img_path = os.path.join(BACKGROUND_PATH, background)

            rotation = round(random.uniform(
                ROTATE_RANGE[0], ROTATE_RANGE[1]), 2)
            translate_x = round(
                random.uniform(-TRANSLATE_RANGE[0], TRANSLATE_RANGE[0]), 2)
            translate_y = round(
                random.uniform(-TRANSLATE_RANGE[1], TRANSLATE_RANGE[1]), 2)
            projection_x = round(random.uniform(
                PROJECTION_RANGE[0], PROJECTION_RANGE[1]), 2)
            projection_y = round(random.uniform(
                PROJECTION_RANGE[0], PROJECTION_RANGE[1]), 2)
            saturation = round(random.uniform(
                SATURATION_RANGE[0], SATURATION_RANGE[1]), 2)
            value = round(random.uniform(VALUE_RANGE[0], VALUE_RANGE[1]), 2)

            print("Rotation: " + str(rotation), "Translate_X: " + str(translate_x), "Translate_Y: " + str(translate_y),
                  "Projection_X " +
                  str(projection_x), "Projection_Y : " + str(projection_y),
                  "Saturation: " + str(saturation), "Value: " + str(value))

            background_img = cv.imread(background_img_path)
            b_dimensions = background_img.shape[:2]

            background_img = resize_no_aspect(
                background_img, TARGET_WIDTH, TARGET_HEIGHT)

            blank = np.zeros(
                shape=(TARGET_HEIGHT, TARGET_WIDTH, 3), dtype='uint8')
            blank[:] = (0, 0, 0)

            # cv.imshow('m', mask)
            img_resized_HSV = changeHSV(img_resized, saturation, value)
            pic = add_obj(blank, img_resized_HSV, mask,
                          TARGET_WIDTH//2, TARGET_HEIGHT//2)
            pic = projectiveTransform(
                pic, rotation, (translate_x, translate_y), (projection_x, projection_y))

            try:
                no_mask, card_info = extract_uno(pic)
                x, y, w, h = card_info
                # print(card_info)

                pic_gray = cv.cvtColor(pic, cv.COLOR_BGR2GRAY)
                ret, pic_mask = cv.threshold(
                    pic_gray, 10, 255, cv.THRESH_BINARY)
                background_mask = cv.bitwise_not(pic_mask)
                card_masked = cv.bitwise_and(pic, pic, mask=pic_mask)
                background_masked = cv.bitwise_and(
                    background_img, background_img, mask=background_mask)
                pic = cv.add(card_masked, background_masked)

                # cv.rectangle(pic, (x, y), (x+w, y+h), (0, 255, 0), 2)

                print(current_output)

                FILENAME = os.path.splitext(
                    background)[0].lower() + '_' + cardType
                CARD_SYMBOL = os.path.splitext(filename)[0].lower()
                FILENAME_YOLOV5 = os.path.splitext(background)[0].lower(
                ) + '_' + CARD_TYPE[index] + ' ' + CARD_SYMBOL + '.txt'

                cv.imwrite(current_output + FILENAME, pic)
                convert_to_yolov5(card_info, CARD_SYMBOL,
                                  FILENAME_YOLOV5, current_label_path)
                # cv.imshow("UNO", pic)
                cv.waitKey(0)
            except:
                save_file_name = os.path.join(OUTPUT, 'file_error')
                FILENAME = os.path.splitext(
                    background)[0].lower() + '_' + cardType
                print(FILENAME, file=open(save_file_name + ".txt", "a"))
                print("An exception occurred")

            cv.destroyAllWindows()
