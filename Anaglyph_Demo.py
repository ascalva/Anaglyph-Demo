import numpy as np
import cv2 as cv

import align as al
from camera_setup import init_cameras

def print_menu():
    print("Command options:\n"
        + "  s - take screenshot\n"
        + "  r - start recording\n"
        + "  a - realign frames\n"
        + "  f - flip left camera\n"
        + "  q - quit\n"
    )

def createVideoWriter(frame, indx):
    frame_width  = frame.shape[1]
    frame_height = frame.shape[0]
    fps          = 30
    vid_name     = "anaglyph_video_{0}.mov".format(indx)

    # Create and return video writer object to save output of program
    return cv.VideoWriter(
                vid_name,
                cv.VideoWriter_fourcc('m','p','4','v'),
                fps,
                (
                    frame_width,
                    frame_height
                ),
                True
            )

def find_smallest_frame(frame_rght, frame_left):

    # Extract dimensions
    row_r,col_r,_ = frame_rght.shape
    row_l,col_l,_ = frame_left.shape

    # Check if the right frame is the smallest
    if (row_r < row_l) or (col_r < col_l) :
        print("The right frame size is smaller")
        return al.RGHT_CAM

    # The left frame is the smallest
    elif (row_l < row_r) or (col_l < col_r) :
        print("The left frame size is smaller")
        return al.LEFT_CAM

    print("Something's not right...")
    return -1


def resize_frame(frame_rght, frame_left, smallest_frame):

    if smallest_frame == al.RGHT_CAM :
        frame_left = cv.resize(frame_left, frame_rght[:2])

    elif smallest_frame == al.LEFT_CAM :
        frame_rght = cv.resize(frame_rght, frame_left[:2])

    # Return both frames where only one has been resized
    return frame_rght, frame_left


def main():
    winName            = 'AnaglyphImage'
    smallest_frame     = 0

    # If the left camera is flipped (to get cameras closer), rotate every frame
    # taken by left camera. In this case, value is True.
    # If value is False, frames are not rotated.
    flip_left_camera   = False

    # Connect to cameras
    rght_cam, left_cam = init_cameras()
    _, frame_rght      = rght_cam.read()
    _, frame_left      = left_cam.read()

    # Check that both cameras are the same shape
    # TODO: Resize frames if they are not the shape
    if frame_rght.shape != frame_left.shape:
        print("Dimensions of right and left camera do not match!")

        # Find the smallest frame
        smallest_frame = find_smallest_frame(frame_rght, frame_left)

        # Exit if something went wrong
        if smallest_frame == -1:
            return

        # Resize the largest frame
        frame_rght,frame_left = resize_frame(frame_rght, frame_left, smallest_frame)

    # Get frame dimensions
    rows,cols,_ = frame_left.shape

    # Create space for an Anaglyph Camera :
    new_ag = np.zeros((rows, cols, 3))

    # Create rotation matrix for left frames if left camera is flipped
    image_center = tuple(np.array(frame_left.shape[1::-1]) / 2)
    M            = cv.getRotationMatrix2D(image_center, 180, 1)

    # Rotate left frame if left camera is flipped
    if flip_left_camera :
        frame_left = cv.warpAffine(frame_left, M, (cols,rows))

    # Compute amout of misalignment between both left and right frames
    y_shift = al.match_frames(frame_rght, frame_left)

    # Get crop indices to crop frames
    top_r,bot_r,top_l,bot_l = al.get_crop_indices(y_shift, rows)

    # Init state variables
    snap  = 0
    rec   = False

    # Print menu
    print_menu()

    # Set up window
    cv.namedWindow(  winName, cv.WINDOW_NORMAL, )
    cv.moveWindow(   winName, 400,  0 )
    cv.resizeWindow( winName, rows, cols )

    while True :

        # Get the next frame from both cameras
        ret_rght, frame_rght_og = rght_cam.read()
        ret_left, frame_left_og = left_cam.read()

        # Resize the largest frame if dimensions do not match
        if smallest_frame > 0:
            frame_rght_og,frame_left_og \
                = resize_frame(frame_rght_og, frame_left_og, smallest_frame)

        # Rotate image if left camera is flipped
        if flip_left_camera :
            frame_left_og = cv.warpAffine(frame_left_og, M, (cols,rows))

        # Crop frames
        frame_rght = frame_rght_og[top_r:bot_r,:]
        frame_left = frame_left_og[top_l:bot_l,:]

        # Check that both succeeded
        if not ret_left or not ret_rght :
            break

        # Split the frames into separate image planes:
        l_blu,l_grn,l_red = cv.split(frame_left)
        r_blu,r_grn,r_red = cv.split(frame_rght)

        # Merge chennels to form anaglyph image
        ana_img           = cv.merge((r_grn,r_grn,l_grn))
        # ana_img           = cv.merge((l_red, r_blu, r_grn))
        # ana_img           = cv.merge((r_blu, r_grn, l_red))

        cv.imshow(winName, ana_img)

        # Save frame to video if recording
        if rec : out.write(ana_img)

        # User input
        c = cv.waitKey(20) & 0xff

        # Quit program
        if c == ord('q') :
            print('\nOkay.  Quiting.  Done.')
            break

        # Save current frame
        elif c == ord("s") :
            print("Saving screenshot..")
            cv.imwrite("anaglyph_output_{0}.png".format(snap), ana_img)
            snap += 1

        # TODO: Start/Stop recording
        elif c == ord("r") :
            if not rec :
                print("Started recording")

            else :
                out.release()
                out = createVideoWriter(ana_img, snap)
                print("Stopped recording")

            rec = not rec

        # Realign cameras (if they were moved)
        elif c == ord("a") :
            print("Realigning cameras..")

            y_shift                 = al.match_frames(frame_rght_og, frame_left_og)
            top_r,bot_r,top_l,bot_l = al.get_crop_indices(y_shift, rows)

        # Start/Stop rotating left frames
        elif c == ord("f") :
            if flip_left_camera : print("Stopped rotating left frames")
            else                : print("Started rotating left frames")

            flip_left_camera    = not flip_left_camera


    # Clean up after yourself:
    rght_cam.release()
    left_cam.release()
    cv.destroyAllWindows()


main()
