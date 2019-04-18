import numpy as np
import cv2 as cv

# Globals
LAPTOP_CAM 	= 0
RGHT_CAM 	= 1
LEFT_CAM 	= 2
WAIT        = 100

def init_cameras():
    attempt = 1
    print("\nAttempt #{0}".format(attempt))

    while True :
        cen_cam            = cv.VideoCapture(LAPTOP_CAM)
        ret_cen, frame_cen = cen_cam.read()
        if ( ret_cen ) :
            print("Center Camera okay.")
            print("Center  image size is: {0} x {1}\n".format(frame_cen.shape[0],frame_cen.shape[1]))
        else :
            print("Center Camera FAILED.")

        cv.waitKey(WAIT)

        rght_cam             = cv.VideoCapture(RGHT_CAM)
        ret_rght, frame_rght = rght_cam.read()
        if ( ret_rght ) :
            print("Right Camera okay.")
            print("Right  image size is: {0} x {1}\n".format(frame_rght.shape[0],frame_rght.shape[1]))
        else :
            print("Right Camera FAILED.")

        cv.waitKey(WAIT)

        left_cam             = cv.VideoCapture(LEFT_CAM)
        ret_left, frame_left = left_cam.read()
        if ( ret_left ) :
            print("Left Camera okay.")
            print("Left  image size is: {0} x {1}\n".format(frame_left.shape[0],frame_left.shape[1]))
        else :
            print("Left Camera FAILED.")

        if ret_cen and ret_rght and ret_left :
            cen_cam.release()

            return rght_cam, left_cam

        else:
            cen_cam .release()
            rght_cam.release()
            left_cam.release()

            attempt += 1
            print("\nAttempt #{0}".format(attempt))


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


def main():
    winName            = 'AnaglyphImage'

    rght_cam, left_cam = init_cameras()
    _, frame_rght      = rght_cam.read()
    _, frame_left      = left_cam.read()

    if frame_rght.shape != frame_left.shape:
        print("Dimensions of right and left camera do not match!")
        return

    # Set up window
    cv.namedWindow(  winName, cv.WINDOW_NORMAL, )
    cv.moveWindow(   winName, 400, 0 )
    cv.resizeWindow( winName, *frame_left.shape[:2] )


    rows = frame_left.shape[0]
    cols = frame_left.shape[1]

    # Create space for an Anaglyph Camera :
    new_ag = np.zeros((rows, cols, 3))

    # Create rotation matrix for left frames if left camera is flipped
    # M = cv.getRotationMatrix2D((cols/2, rows/2), 180, 1)

    y_shift = match_frames(frame_rght, frame_left)

    # Get crop indices to crop frames
    top_r,bot_r,top_l,bot_l = get_crop_indices(y_shift, rows)

    snap  = 0
    rec   = False

    while True :

        # Get the next frame from both cameras
        ret_rght, frame_rght_og = rght_cam.read()
        ret_left, frame_left_og = left_cam.read()

        # Crop frames
        frame_rght = frame_rght_og[top_r:bot_r,:]
        frame_left = frame_left_og[top_l:bot_l,:]

        # Check that both succeeded
        if not ret_left or not ret_rght:
            break

        # Rotate image if left camera is flipped
        # frame_left = cv.warpAffine(frame_left, M, (cols,rows))

        # Split the frames into separate image planes:
        l_blu,l_grn,l_red = cv.split(frame_left)
        r_blu,r_grn,r_red = cv.split(frame_rght)
        ana_img           = cv.merge((r_grn,r_grn,l_grn))

        cv.imshow(winName, ana_img)

        # User input
        c = cv.waitKey(20) & 0xff
        if c == ord('q') :
            print('\nOkay.  Quiting.  Done.')
            break

        elif c == ord("s") :
            print("Saving screenshot..")
            cv.imwrite("anaglyph_output_{0}.png".format(snap), ana_img)
            snap += 1

        elif c == ord("r") :
            if not rec : print("Started recording")
            else       : print("Stopped recording")

            rec        = not rec

        elif c == ord("a"):
            print("Realigning cameras..")

            y_shift                 = match_frames(frame_rght_og, frame_left_og)
            top_r,bot_r,top_l,bot_l = get_crop_indices(y_shift, rows)


    # Clean up after yourself:
    rght_cam.release()
    left_cam.release()
    cv.destroyAllWindows()


main()
