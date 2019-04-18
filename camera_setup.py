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
