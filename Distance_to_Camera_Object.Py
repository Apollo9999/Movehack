# import the necessary packages
 
from imutils.video import VideoStream
from imutils import paths
import argparse
import datetime
import imutils
import time
import cv2
 
# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=500, help="minimum area size")
args = vars(ap.parse_args())
 
# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
 
# otherwise, we are reading from a video file
else:
    vs = cv2.VideoCapture(args["video"])
 
# initialize the first frame in the video stream
firstFrame = None
#################################################################################################
def distance_to_camera(knownWidth, focalLength, perWidth):
        # compute and return the distance from the maker to the camera
        return (knownWidth * focalLength) / perWidth
 
 
def find_marker(image):
        # convert the image to grayscale, blur it, and detect edges
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)
 
        # find the contours in the edged image and keep the largest one;
        # we'll assume that this is our piece of paper in the image
        cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        c = max(cnts, key = cv2.contourArea)
 
        # compute the bounding box of the of the paper region and return it
        return cv2.minAreaRect(c)
# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 24.0
 
# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 11.0
 
 
#################################################################################################
       
# loop over the frames of the video
while True:
        # grab the current frame and initialize the occupied/unoccupied
        # text
        frame = vs.read()
        frame = frame if args.get("video", None) is None else frame[1]
        text = "Unoccupied"
 
        # if the frame could not be grabbed, then we have reached the end
        # of the video
        if frame is None:
                break
 
        # resize the frame, convert it to grayscale, and blur it
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
 
       
        marker = find_marker(frame)
        focalLength = (marker[1][0] * KNOWN_DISTANCE) / KNOWN_WIDTH
#################################################################################
# loop over the images
        for imagePath in sorted(paths.list_images("images")):
                # load the image, find the marker in the image, then compute the
                # distance to the marker from the camera
                image = cv2.imread(imagePath)
                marker = find_marker(image)
                inches = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
 
                # draw a bounding box around the image and display it
                box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
                box = np.int0(box)
                cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
                cv2.putText(image, "%.2fft" % (inches / 12),
                        (image.shape[1] - 200, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (0, 255, 0), 3)
                cv2.imshow("image", image)
##################################################################################
        # if the first frame is None, initialize it
        if firstFrame is None:
                firstFrame = gray
                continue
               
        # compute the absolute difference between the current frame and
        # first frame
        frameDelta = cv2.absdiff(firstFrame, gray)
        thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
 
        # dilate the thresholded image to fill in holes, then find contours
        # on thresholded image
        thresh = cv2.dilate(thresh, None, iterations=2)
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
 
        # loop over the contours
        for c in cnts:
                # if the contour is too small, ignore it
                if cv2.contourArea(c) < args["min_area"]:
                        continue
 
                # compute the bounding box for the contour, draw it on the frame,
                # and update the text
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                text = "Occupied"
 
        # draw the text and timestamp on the frame
        cv2.putText(frame, "Room Status: {}".format(text), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
 
        # show the frame and record if the user presses a key
        cv2.imshow("Security Feed", frame)
        cv2.imshow("Thresh", thresh)
        cv2.imshow("Frame Delta", frameDelta)
        key = cv2.waitKey(1) & 0xFF
 
        # if the `q` key is pressed, break from the lop
        if key == ord("q"):
                break
 
# cleanup the camera and close any open windows
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()
