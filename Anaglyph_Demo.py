import numpy as np
import cv2 as cv

from align import match_frames, get_crop_indices
from camera_setup import init_cameras

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
