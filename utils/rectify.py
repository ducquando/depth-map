import numpy as np
import cv2 as cv
import yaml

def get_cam_params(path):
    """
    Get camera's instrinsic matrix and distortion coefficients
    """
    # Read file
    with open(path) as file:
        mtx = yaml.safe_load(file)
        
    # Get instrinsic matrix
    intrinsic = np.array(mtx["camera_matrix"])
    
    # Get distortion coefficients
    distort_coeff = np.array(mtx["distort_coeff"])

    return intrinsic, distort_coeff
    
    
def transform_mtx(iframe, path = ""):
    """
    Find transformation matrix between the input frame and the referenced image
    """
    reference = cv.imread(path + 'assets/rectify/book_background.jpg', cv.IMREAD_GRAYSCALE)

    # Initiate SIFT detector
    sift = cv.SIFT_create()
    MIN_MATCH_COUNT, FLANN_INDEX_KDTREE = 10, 1

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(reference, None)
    kp2, des2 = sift.detectAndCompute(iframe, None)
    
    # Find matches
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k = 2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
            
    if len(good) < MIN_MATCH_COUNT:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        
    # Calculate transformation matrix
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    mtx, _ = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)

    return mtx
    
def homo_translation(mtx):
    """
    Convert matrix into homogenious matrix
    """
    homo = np.identity(4)
    for i in range(mtx.shape[0]):
        homo[i][3] = mtx[i][0]
        
    return homo
    
def rectify(**kwargs):
    """
    Get parameters to rectify images
    """
    # Get kwargs
    path = kwargs.get('path', "")
    imgL = kwargs.get('imgL', cv.imread(path + 'assets/mono/position1/left.jpg', cv.COLOR_BGR2GRAY))
    imgR = kwargs.get('imgR', cv.imread(path + 'assets/mono/position1/right.jpg', cv.COLOR_BGR2GRAY))
    
    # Get cameras' instrinsic and extrinsic matrices
    mtx, distort = get_cam_params(path + 'outputs/left.yaml')
    extL, extR = transform_mtx(imgL, path), transform_mtx(imgR, path)
    
    # Get rotation and translation matrices
    _, _, transL, _ = cv.decomposeHomographyMat(extL, mtx)
    _, _, transR, _ = cv.decomposeHomographyMat(extR, mtx)
    transL, transR = homo_translation(transL[2]), homo_translation(transR[2])
    
    # Transformation from right to left camera
    trans_RtoL = np.dot(np.linalg.inv(transR), transL)[:3,0]
    rot_RtoL = np.identity(3)

    # Rectify
    rR, rL, prjR, prjL, _, _, _ = cv.stereoRectify(mtx, distort, mtx, distort,
                                                   (imgR.shape[1], imgR.shape[0]), rot_RtoL, trans_RtoL, alpha = -1)
    
    # Store everything in a dictionary
    results = {"mtx": mtx, "distort": distort, "rectifyL": rL, "rectifyR": rR, "prjL": prjL, "prjR": prjR}

    return results
    

def undistort(imgL, imgR, rectified):
    """
    Undistort left and right images
    """
    # Get params
    mtx, distort = rectified["mtx"], rectified["distort"]
    prjL, prjR, rectifyL, rectifyR = rectified["prjL"], rectified["prjR"], rectified["rectifyL"], rectified["rectifyR"]
    
    # Undistort the images
    mapL1, mapL2 = cv.initUndistortRectifyMap(mtx, distort, rectifyL, prjL, (imgL.shape[1], imgL.shape[0]), cv.CV_32FC1)
    mapR1, mapR2 = cv.initUndistortRectifyMap(mtx, distort, rectifyR, prjR, (imgR.shape[1], imgR.shape[0]), cv.CV_32FC1)
    imgL_rect = cv.remap(imgL, mapL1, mapL2, cv.INTER_LINEAR)
    imgR_rect = cv.remap(imgR, mapR1, mapR2, cv.INTER_LINEAR)
    
    # Crop and resize images
    imgL_roi = cv.resize(imgL_rect[:603,:1072], (1280, 720))
    imgR_roi = cv.resize(imgR_rect[:603,:1072], (1280, 720))

    return imgL_roi, imgR_roi