import cv2 as cv
import numpy as np
import pickle
from cv2 import aruco
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Load the camera matrix and distortion coefficients
with open("cameraMatrix.pkl", "rb") as f:
    cameraMatrix = pickle.load(f)

with open("dist.pkl", "rb") as f:
    distCoeffs = pickle.load(f)

# Define the ArUco marker size (for example, 5 cm x 5 cm)
markerLength = 0.095  # In meters (5 cm)

# Define the ArUco dictionary (choose the one that matches your markers)
arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_50)
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

        # Draw the marker axes and the ID on the frame
        cv.putText(frame, str(ids[0][0]), (int(corners[0][0][0][0]), int(corners[0][0][0][1])), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        
        # Calculate and display the distance
        distance = np.linalg.norm(tvec[0][0])
        cv.putText(frame, f"Dist: {distance:.2f}m", (int(corners[0][0][0][0]), int(corners[0][0][0][1])+15), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv.LINE_AA)
        projected_point_3D = np.array([(tvec[0][0][0], tvec[0][0][1], 0)], dtype=np.float32).reshape(-1, 3)
        projected_point_2D, _ = cv.projectPoints(projected_point_3D, rvec, tvec, cameraMatrix, distCoeffs)
        origin_2d, _ = cv.projectPoints((0,0,0), rvec, tvec, cameraMatrix, distCoeffs)
        projected_pixel = tuple(projected_point_2D[0][0].astype(int))
        origin = tuple(origin_2d[0][0].astype(int))
        cv.line(frame, projected_pixel, (360          ,0),(0,255,0),3)
        cv.line(frame, (int(corners[0][0][0][0]), int(corners[0][0][0][1])),projected_pixel,(0,255,0),3)
        cv.circle(frame, (int(corners[0][0][0][0]), int(corners[0][0][0][1])), 5, (255, 0, 0), -1)  # Blue circles

    # Display the frame
    cv.imshow('Frame', frame)

    # Exit if 'q' is pressed
    if cv.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv.destroyAllWindows()


# Assuming tvec represents the translation of the ArUco marker from the camera in world coordinates
marker_point_3D = np.array([tvec[0][0]], dtype=np.float32).reshape(-1, 3)
# Projected point on the XY plane (assuming Z=0 for the projection)
projected_point_3D = np.array([(tvec[0][0][0], tvec[0][0][1], 0)], dtype=np.float32).reshape(-1, 3)

# Assuming the origin is at (0,0,0) in 3D world coordinates
origin_3D = np.array([0, 0, 0], dtype=np.float32)

# Correct the interpolate_waypoints function if necessary to handle 3D points
def interpolate_waypoints(start, end, step):
    distance = np.linalg.norm(end - start)
    steps = int(distance / step)
    direction = (end - start) / distance
    return [start + direction * step * i for i in range(steps + 1)]

# Use the corrected 3D points for interpolation
waypoints_origin_to_projected = interpolate_waypoints(origin_3D, projected_point_3D[0], 0.1)
waypoints_projected_to_marker = interpolate_waypoints(projected_point_3D[0], marker_point_3D[0], 0.1)

# Follow with your plotting logic...
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot waypoints
for point in waypoints_origin_to_projected:
    ax.scatter(*point, color='blue')
for point in waypoints_projected_to_marker:
    ax.scatter(*point, color='red')

# Plot lines for visual aid
ax.plot(*zip(origin_3D, projected_point_3D[0]), color='blue', linestyle='dashed')
ax.plot(*zip(projected_point_3D[0], marker_point_3D[0]), color='red', linestyle='dashed')

# Annotate the marker point with its coordinates in green color
marker_label = f"Aruco Marker ({marker_point_3D[0][0]:.2f}, {marker_point_3D[0][1]:.2f}, {marker_point_3D[0][2]:.2f})"
ax.scatter(*marker_point_3D[0], color='green')  # Plot the marker point in green
ax.text(marker_point_3D[0][0], marker_point_3D[0][1], marker_point_3D[0][2], marker_label, color='green')

# Setting labels
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax_2d = fig.add_subplot(122)
# Convert the last frame from BGR to RGB (matplotlib expects RGB format)
frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
ax_2d.imshow(frame_rgb)
ax_2d.axis('off')  # Hide axis
ax_2d.set_title('Last Captured Frame')


plt.show()


