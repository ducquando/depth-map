import numpy as np
import cv2 as cv
import pandas as pd
import yaml

# Allow realtime.py to run on another device that haven't install pyntcloud 
try:
    from pyntcloud import PyntCloud
except ImportError:
    pass

def return_depth(disparity, stereo, this_path = ""):
    """
    Calculate depth map knowing the disparity map, the camera's focal lens, and the baseline.
    Thanks to maths, we can also use this function to calculate disparity map knowing the rest.
    """
    BASELINE = 9 if stereo else 5 # cm
    FOCAL_LENGTH = get_focal_px(this_path + 'outputs/left.yaml')  # px/px
    
    # Cast type of disparity
    disparity = float(disparity) if type(disparity) == int else disparity
    
    # Calculate depth map
    depth = np.divide(BASELINE * FOCAL_LENGTH, disparity, out = np.zeros_like(disparity), where = disparity != 0.0)

    return depth  # cm

def get_focal_px(path):
    # Read file
    with open(path) as file:
        mtx = yaml.safe_load(file)
        
    # Get instrinsic matrix
    return float(mtx["focal_cm"])

def px_cm_ratio(depth, stereo = True, from_cm = True):
    """
    Convert to pixel to cm, and vice versa
    """
    # Known matrices
    single = [[50, 50, 85],
              [3.86232965, 3.72734804, 6.42447434]]
    twice = [[50, 70, 100],
             [4.72974624, 6.67173753, 9.98931149]]
    
    # Cast x, y according to inputs
    mtx = twice if stereo else single
    (y, x) = (mtx[0], mtx[1]) if from_cm else (mtx[1], mtx[0])
    
    # Calculate the scalar
    A = np.vstack([y, np.ones(len(y))]).T
    m, c = np.linalg.lstsq(A, x, rcond=-1)[0]
        
    # Return y = mx + c
    return m * depth + c


def depth_to_ply(image, depth, name = "my_pts"):
    """
    Convert depth map to point cloud
    """
    # Extract the coordinates of the points where there is actual values (not NaN) in the xyz cloud of points
    points_rows, points_cols = np.where(~np.isnan(depth[:, :, 0]))
    
    # Grab the corresponding points in the xyz cloud of points in an array
    points_depth = depth[points_rows, points_cols, :] # n*3 array of 3D points (after nan filtering)
    
    # Grab the corresponding points in the image in an array
    points_image =  image[points_rows, points_cols, 0:3] # n*3 array of RGB points (after nan filtering)
    
    # Create a dict of data
    data = {'x': points_depth[:, 0], 'y': points_depth[:, 1], 'z': points_depth[:, 2], 
            'red': points_image[:, 2], 'green': points_image[:, 1], 'blue': points_image[:, 0]}
            
    # build a cloud
    cloud = PyntCloud(pd.DataFrame(data))

    # Write .ply file
    cloud.to_file('outputs/' + name + '.ply')
    
    return cloud, data


def display_individual_depth(dispar_map, dispar_range, stereo, debug = False):
    """
    Return the depth of the object at that returned location with a pre-specified depth range
    """
    # Remove out-of-bound area
    mask = cv.inRange(dispar_map, dispar_range[0], dispar_range[1])
    dispar_safe = cv.bitwise_and(dispar_map, dispar_map, mask = mask)
    
    # Calculate depth map
    depth_map = return_depth(dispar_safe, stereo, "../") if debug else px_cm_ratio(return_depth(dispar_safe, stereo), stereo, False)
    
    # Contour detection 
    contours, _ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cnts = sorted(contours, key = cv.contourArea, reverse = True)

    # Check if detected contour is significantly large (to avoid multiple tiny regions)
    x, y, w, h = cv.boundingRect(cnts[0])

    # Finding average depth of region represented by the largest contour 
    new_mask = np.zeros_like(mask)
    cv.drawContours(new_mask, cnts, 0, (255), -1)

    # Calculating the average depth of the object closer than the safe distance
    depth_mean, _ = cv.meanStdDev(depth_map, mask = new_mask)
    
    return x, y, x + w, y + h, depth_mean
