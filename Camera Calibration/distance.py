import cv2 as cv
import numpy as np
import pickle
from cv2 import aruco

# Load the camera matrix and distortion coefficients
with open("cameraMatrix.pkl", "rb") as f:
    cameraMatrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    distCoeffs = pickle.load(f)

# Define the ArUco marker size (for example, 5 cm x 5 cm)
markerLength = 0.095  # In meters (5 cm)

# Define the ArUco dictionary (choose the one that matches your markers)
arucoDict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
arucoParams = aruco.DetectorParameters_create()

# Start video capture
cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Detect ArUco markers
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, arucoDict, parameters=arucoParams)

    # If markers are detected, process only the first detected marker
    if ids is not None and len(ids) > 0:
        # Estimate pose of the first marker
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners[0], markerLength, cameraMatrix, distCoeffs)
        R, _ = cv.Rodrigues(rvec)
        # Invert the rotation matrix
        R_inv = np.transpose(R)
        # Invert the translation vector
        tvec_inv = -np.dot(R_inv, tvec[0][0].reshape((3, 1)))

        # Draw the marker axes and the ID on the frame
        cv.putText(frame, str(ids[0][0]), (int(corners[0][0][0][0]), int(corners[0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        
        # Calculate and display the distance
        distance = np.linalg.norm(tvec[0][0])
        cv.putText(frame, f"Dist: {distance:.2f}m", (int(corners[0][0][0][0]), int(corners[0][0][0][1])+15), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        print(tvec)


    # Display the frame
    cv.imshow('Frame', frame)


    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv.destroyAllWindows()
