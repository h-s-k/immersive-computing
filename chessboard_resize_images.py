import os
import cv2
import numpy as np

# Parameters
folder_path = "chessboard_calibration_data"
n = 50  # width resize percentage
m = 50  # height resize percentage
chessboard_size = (9, 6)  # size of your chessboard pattern (number of inner corners)

# Initialize lists to store object points and image points
# 3D points in real world space
objpoints = []
# 2D points in image plane
imgpoints = []

# Prepare object points (0, 0, 0), (1, 0, 0), (2, 0, 0), ..., (8, 5, 0)
objp = np.zeros((chessboard_size[1] * chessboard_size[0], 3), np.float32)
objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)

# Read images from folder
for filename in os.listdir(folder_path):
    print(filename)
    if filename.endswith(".jpeg"):
        img_path = os.path.join(folder_path, filename)
        img = cv2.imread(img_path)
        
        # Resize image
        width = int(img.shape[1] * n / 100)
        height = int(img.shape[0] * m / 100)
        dim = (width, height)
        resized_img = cv2.resize(img, dim)

        # Convert to grayscale
        gray = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
        print(gray)

        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)

        # If found, add object points, image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            cv2.drawChessboardCorners(resized_img, chessboard_size, corners, ret)
            cv2.imshow("Chessboard", resized_img)
            cv2.waitKey(500)

cv2.destroyAllWindows()