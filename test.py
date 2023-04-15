import dlib
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

# initialize Dlib's face detector and facial landmark detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    # load the image
    # image = cv2.imread("example.jpg")
    _, frame = cap.read()

    # convert the image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image using Dlib's face detector
    faces = detector(gray, 1)

    # loop over each face and extract the eye regions
    for face in faces:
        # detect the facial landmarks for the face using Dlib's facial landmark detector
        landmarks = predictor(gray, face)
        
        # extract the coordinates of the left and right eye regions
        left_eye_points = np.array([(landmarks.part(36).x, landmarks.part(36).y),
                                    (landmarks.part(37).x, landmarks.part(37).y),
                                    (landmarks.part(38).x, landmarks.part(38).y),
                                    (landmarks.part(39).x, landmarks.part(39).y),
                                    (landmarks.part(40).x, landmarks.part(40).y),
                                    (landmarks.part(41).x, landmarks.part(41).y)], np.int32)
        
        right_eye_points = np.array([(landmarks.part(42).x, landmarks.part(42).y),
                                    (landmarks.part(43).x, landmarks.part(43).y),
                                    (landmarks.part(44).x, landmarks.part(44).y),
                                    (landmarks.part(45).x, landmarks.part(45).y),
                                    (landmarks.part(46).x, landmarks.part(46).y),
                                    (landmarks.part(47).x, landmarks.part(47).y)], np.int32)
        
        # extract the eye regions using OpenCV's fillConvexPoly function
        # mask = np.zeros_like(gray)
        # cv2.fillConvexPoly(mask, left_eye_points, 255)
        # left_eye = cv2.bitwise_and(gray, mask)
        
        # mask = np.zeros_like(gray)
        # cv2.fillConvexPoly(mask, right_eye_points, 255)
        # right_eye = cv2.bitwise_and(gray, mask)

        mask = np.zeros_like(gray)
        cv2.fillConvexPoly(mask, left_eye_points, 255)
        cv2.fillConvexPoly(mask, right_eye_points, 255)
        both_eye = cv2.bitwise_and(gray, mask)
            
        # display the eye regions
        # cv2.imshow("Left Eye", left_eye)
        # cv2.imshow("Right Eye", right_eye)
        cv2.imshow("Both Eye", both_eye)

        # # Detect the edges of the eye using the Canny edge detector
        # edges = cv2.Canny(gray, 100, 200)

        # # Detect the corners of the eye using the Harris corner detector
        # corners = cv2.cornerHarris(gray, 2, 3, 0.04)

        # # Compute the shape of the eye using the Hough transform
        # circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
        #                         param1=50, param2=30, minRadius=0, maxRadius=0)

        # # Compute the size of the pupil using the threshold function
        # ret, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
        # contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # if len(contours) > 0:
        #     c = max(contours, key=cv2.contourArea)
        #     ((x, y), r) = cv2.minEnclosingCircle(c)
        #     pupil_size = 2 * r

        # # Print out the features
        # print("Number of edges:", len(edges))
        # print("Number of corners:", len(corners))
        # print("Eye shape:", circles)
        # print("Pupil size:", pupil_size)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()