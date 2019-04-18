import cv2 as cv

def drawMatches(frame_rght, kp1, frame_left, kp2, matches):

    avg_x_rght = 0
    avg_x_left = 0
    avg_y_rght = 0
    avg_y_left = 0
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for mat in matches:

        # Get the matching keypoints for each of the images
        img1_idx = mat.queryIdx
        img2_idx = mat.trainIdx

        # x - columns
        # y - rows
        (x1,y1) = kp1[img1_idx].pt
        (x2,y2) = kp2[img2_idx].pt

        avg_x_rght += x1
        avg_x_left += x2
        avg_y_rght += y1
        avg_y_left += y2

    avg_x_rght /= len(matches)
    avg_x_left /= len(matches)
    avg_y_rght /= len(matches)
    avg_y_left /= len(matches)

    return avg_y_rght - avg_y_left


def match_frames(frame_rght, frame_left):
    # ORB detector: 1000 keypoints; scaling pyramid factor of 1.2
    orb       = cv.ORB_create()#1000, 1.2)

    # Detect keypoints of original image
    kp_r, des_r = orb.detectAndCompute(frame_rght, None)
    kp_l, des_l = orb.detectAndCompute(frame_left, None)

    # Create matcher
    bf        = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches   = bf.match(des_r, des_l)
    matches   = sorted(matches, key=lambda val: val.distance) # based on distance

    numGoodMatches = int(len(matches) * 0.15)

    # out     = np.zeros((max([rows1,rows2]), cols1+cols2, 3), dtype='uint8')
    y_shift = drawMatches(frame_rght, kp_r, frame_left, kp_l, matches[:numGoodMatches])

    return int(y_shift)


def get_crop_indices(y_shift, height):
    top_l = 0
    bot_l = height
    top_r = 0
    bot_r = height

    # Right image is lower
    if y_shift < 0:
        y_shift = abs(y_shift)
        top_l   = y_shift
        bot_l   = height
        top_r   = 0
        bot_r   = height - y_shift

    # Left image is lower
    elif y_shift > 0:
        y_shift = abs(y_shift)
        top_l   = 0
        bot_l   = height - y_shift
        top_r   = y_shift
        bot_r   = height

    return top_r, bot_r, top_l, bot_l
