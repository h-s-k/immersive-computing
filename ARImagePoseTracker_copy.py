import cv2
import numpy as np
import os
import CalibrationHelpers as calib

# This function is yours to complete
# it should take in a set of 3d points and the intrinsic matrix
# rotation matrix(R) and translation vector(T) of a camera
# it should return the 2d projection of the 3d points onto the camera defined
# by the input parameters 
def ProjectPoints(points3d, new_intrinsics, R, T):

    # Convert 3D points to homogeneous coordinates
    points3d_homogeneous = np.hstack((points3d, np.ones((points3d.shape[0], 1))))

    # Create a 4x4 transformation matrix combining rotation and translation
    transformation_matrix = np.eye(4)
    # Set the rotation 
    transformation_matrix[:3, :3] = R 
    # Set the translation
    transformation_matrix[:3, 3] = T

    # Apply the transformation matrix
    points3d_transformed = np.dot(transformation_matrix, points3d_homogeneous.T).T

    # Project 3D points to 2D using new_intrinsics matrix
    points2d_homogeneous = np.dot(new_intrinsics, points3d_transformed[:, :3].T).T

    # Normalize homogeneous coordinates
    points2d = points2d_homogeneous[:, :2] / points2d_homogeneous[:, 2:]

    return points2d

# This function will render a cube on an image whose camera is defined
# by the input intrinsics matrix, rotation matrix(R), and translation vector(T)
def renderCube(img_in, new_intrinsics, R, T):
    # Setup output image
    img = np.copy(img_in)

    # We can define a 10cm cube by 4 sets of 3d points
    # these points are in the reference coordinate frame
    scale = 0.1
    face1 = np.array([[0,0,0],[0,0,scale],[0,scale,scale],[0,scale,0]],
                     np.float32)
    
    face2 = np.array([[0,0,0],[0,scale,0],[scale,scale,0],[scale,0,0]],
                     np.float32)
    face3 = np.array([[0,0,scale],[0,scale,scale],[scale,scale,scale],
                      [scale,0,scale]],np.float32)
    
    face4 = np.array([[scale,0,0],[scale,0,scale],[scale,scale,scale],
                      [scale,scale,0]],np.float32)
    
    # using the function you wrote above we will get the 2d projected 
    # position of these points
    face1_proj = ProjectPoints(face1, new_intrinsics, R, T)

    # this function draws a line connecting the 4 points
    img = cv2.polylines(img, [np.int32(face1_proj)], True, 
                              tuple([255,0,0]), 3, cv2.LINE_AA) 
    
    face2_proj = ProjectPoints(face2, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face2_proj)], True, 
                              tuple([0,255,0]), 3, cv2.LINE_AA) 
    
    face3_proj = ProjectPoints(face3, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face3_proj)], True, 
                              tuple([0,0,255]), 3, cv2.LINE_AA) 
    
    face4_proj = ProjectPoints(face4, new_intrinsics, R, T)
    img = cv2.polylines(img, [np.int32(face4_proj)], True, 
                              tuple([125,125,0]), 3, cv2.LINE_AA) 
    return img

# Function to compute the pose from homography
def ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints):
    # compute homography using RANSAC, this allows us to compute
    # the homography even when some matches are incorrect
    homography, mask = cv2.findHomography(referencePoints, imagePoints, cv2.RANSAC, 5.0)

    # check that enough matches are correct for a reasonable estimate
    MIN_INLIERS = 30
    if sum(mask) > MIN_INLIERS:
        # given that we have a good estimate
        # decompose the homography into Rotation and translation
        RT = np.matmul(np.linalg.inv(new_intrinsics), homography)
        norm = np.sqrt(np.linalg.norm(RT[:, 0]) * np.linalg.norm(RT[:, 1]))
        RT = -1 * RT / norm
        c1 = RT[:, 0]
        c2 = RT[:, 1]
        c3 = np.cross(c1, c2)
        T = RT[:, 2]
        R = np.vstack((c1, c2, c3)).T
        W, U, Vt = cv2.SVDecomp(R)
        R = np.matmul(U, Vt)
        return True, R, T
    # return false if we could not compute a good estimate
    return False, None, None

# Load the reference image that we will try to detect in the webcam
reference = cv2.imread('/Users/helenakent/Documents/ImmersiveComputing/hw4-ar-tracking-h-s-k-revised/box_images/ARTrackerImage.jpg')
RES = 480
reference = cv2.resize(reference, (RES, RES))

# create the feature detector. This will be used to find and describe locations
# in the image that we can reliably detect in multiple images
feature_detector = cv2.BRISK_create(octaves=5)

# compute the features in the reference image
reference_keypoints, reference_descriptors = feature_detector.detectAndCompute(reference, None)

# create the matcher that is used to compare feature similarity
# Brisk descriptors are binary descriptors (a vector of zeros and 1s)
# Thus hamming distance is a good measure of similarity
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# Specify the directory containing the images
image_directory = '/Users/helenakent/Documents/ImmersiveComputing/hw4-ar-tracking-h-s-k-revised/box_images'

# Get a list of image files in the directory
image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.jpeg')]

# Load the camera calibration matrix
intrinsics, distortion, new_intrinsics, roi = calib.LoadCalibrationData('calibration_data')

# Loop through each image file and process them
for image_file in image_files:
    # Load the current image
    current_frame = cv2.imread(image_file)
    current_frame = cv2.resize(current_frame, (RES, RES))
    current_frame = cv2.undistort(current_frame, intrinsics, distortion, None, new_intrinsics)
    x, y, w, h = roi
    current_frame = current_frame[y:y+h, x:x+w]

    # detect features in the current image
    current_keypoints, current_descriptors = feature_detector.detectAndCompute(current_frame, None)

    # match the features from the reference image to the current image
    matches = matcher.match(reference_descriptors, current_descriptors)

    # set up reference points and image points
    referencePoints = np.float32([reference_keypoints[m.queryIdx].pt for m in matches])
    SCALE = 0.1  # scale of the reference image: 0.1m x 0.1m
    referencePoints = SCALE * referencePoints / RES
    imagePoints = np.float32([current_keypoints[m.trainIdx].pt for m in matches])

    # compute homography
    ret, R, T = ComputePoseFromHomography(new_intrinsics, referencePoints, imagePoints)

    render_frame = current_frame
    if ret:
        # compute the projection and render the cube
        render_frame = renderCube(current_frame, new_intrinsics, R, T)

    # display the current image frame with the augmented reality cube
    cv2.imshow('Box Images with Cube', render_frame)

    # Wait for a key event and check if the user pressed 'q' or 'Esc' to exit
    k = cv2.waitKey(0)
    if k == 27 or k == 113:  # 27, 113 are ASCII for escape and q respectively
        # exit
        break

cv2.destroyAllWindows()