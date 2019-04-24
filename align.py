import cv2 as cv

def compute_shift(frame_rght, kp1, frame_left, kp2, matches):
    """
    Use the best matching keypoints between both frames to compute the mean of
    all y-values of all points in each image. Find difference between centers
    """
    avg_x_rght = 0
    avg_x_left = 0
    avg_y_rght = 0
    avg_y_left = 0

    # Iterate over all keypoints to compute the mean in every direction for
    # both frames
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

    # Compute averages
    avg_x_rght /= len(matches)
    avg_x_left /= len(matches)
    avg_y_rght /= len(matches)
    avg_y_left /= len(matches)

    return int(avg_y_rght - avg_y_left)


def match_frames(frame_rght, frame_left):
    """
    Creates unique feature points on both left and right images using ORB and
    computes the amount of shift required to align frames.
    Returns the amount of vertical shift to align both frames.
    """

    # ORB detector: 1000 keypoints; scaling pyramid factor of 1.2
    orb       = cv.ORB_create()#1000, 1.2)

    # Detect keypoints of original image
    kp_r, des_r = orb.detectAndCompute(frame_rght, None)
    kp_l, des_l = orb.detectAndCompute(frame_left, None)

    # Create matcher
    bf        = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches   = bf.match(des_r, des_l)
    matches   = sorted(matches, key=lambda val: val.distance) # based on distance

    # Use the best 15% of matches
    numGoodMatches = int(len(matches) * 0.15)

    # Compute the amount of vertical shift needed to align frames horizontally
    # and return
    return compute_shift(frame_rght, kp_r, frame_left, kp_l, matches[:numGoodMatches])


def get_crop_indices(y_shift, height):
    """
    Compute row indices for both frames using the amount of vertical shift and
    the height of both frames.
    """
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
