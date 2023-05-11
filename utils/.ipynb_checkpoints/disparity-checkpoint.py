import numpy as np
import cv2 as cv

def scale(value, min_dispar, num_dispar):
    """
    Scale the disparity from pixels to pixel/pixel
    """
    return (value / (16 ** 2) - min_dispar) / num_dispar  #cm/cm


def disparity_helper(imgL, imgR, params):
    """
    Return the disparity accordingly to the input images and parameters
    """
    # Get parameters
    kernel, min_dispar, dispar, block = params["kernel"], params["min_disparity"], params["disparity"], params["block"]
    
    # Blur images to remove noises
    blur_imgL = cv.GaussianBlur(imgL, (kernel, kernel), 0)
    blur_imgR = cv.GaussianBlur(imgR, (kernel, kernel), 0)
    
    # Calculate disparity
    stereo = cv.StereoSGBM_create(
        minDisparity = 16 * min_dispar,
        numDisparities = 16 * dispar,
        blockSize = block,
        speckleRange = 2,
        P1 = 8 * 3 * kernel ** 2, P2 = 32 * 3 * kernel ** 2)

    # Return disparity
    return stereo.compute(blur_imgL, blur_imgR)


def calculate_disparity_dim(imgL, imgR, params):
    """
    Calculate the disparity given a specific dimension and parameters
    """
    # Deep copy
    imgL, imgR = imgL.copy(), imgR.copy()
    
    # Resize
    dim = params["dimension"]
    resized_imgL, resized_imgR = cv.resize(imgL, dim), cv.resize(imgR, dim) # Resize
    
    # Calculate disparity and return
    return disparity_helper(resized_imgL, resized_imgR, params)


def combine_disparity(dim, low, mid, high, value_ranges):
    """
    Combine the disparity of different resolutions
    """
    combined = low.copy()
    
    for value_range in value_ranges:
        for i in range(dim[1]):
            for j in range(dim[0]): 
                low_value, mid_value, high_value = low[i][j], mid[i][j], high[i][j]
                if low_value > value_range[0] and low_value < value_range[1]:
                    combined[i][j] = (5 * low_value + 3 * mid_value + 2 * high_value) / 10
    
    return combined


def return_ranges(dispar_map, min_dispar, max_dispar, bins, threshold):
    """
    Return all ranges with intensity larger then the specified threshold
    """
    # Remove the out-of-range regions and make a historgram
    mask = cv.inRange(dispar_map, min_dispar, max_dispar)
    dispar_safe = cv.bitwise_and(dispar_map, dispar_map, mask = mask)
    dispar_hist = np.histogram(dispar_safe, bins, (min_dispar, max_dispar))
    
    # Calculate bin size
    bin_size = (max_dispar - min_dispar) / bins
    
    # Return all ranges that has intensity larger then the threshold
    dispar_range = []
    for i in range(0, bins):
        if dispar_hist[0][i] > threshold:
            dispar_range.append([bin_size * i, bin_size * (i + 1)])
    
    # Add min value to that range
    dispar_range = np.array(dispar_range) + min_dispar
    
    return dispar_range


def return_disparity(imgL, imgR):
    """
    Calculate the disparity of the input images
    """
    # Original dimension
    dim = imgL.shape[::-1]
    
    # 720p
    hi_params = {"dimension": (1280, 720), "kernel": 3, "block": 20,
              "min_disparity": 5, "disparity": 8}
    hi_disparity = calculate_disparity_dim(imgL, imgR, hi_params)
    
    # 360p
    mid_params = {"dimension": (640, 360), "kernel": 5, "block": 18,
              "min_disparity": 0, "disparity": 8}
    mid_disparity = calculate_disparity_dim(imgL, imgR, mid_params)
    
    # 180p
    low_params = {"dimension": (320, 180), "kernel": 7, "block": 15,
              "min_disparity": 0, "disparity": 5}
    low_disparity = calculate_disparity_dim(imgL, imgR, low_params)
    
    # Resize and normalize
    high = scale(cv.resize(hi_disparity, dim), hi_params["min_disparity"], hi_params["disparity"])
    mid = scale(cv.resize(mid_disparity, dim), mid_params["min_disparity"], mid_params["disparity"])
    low = scale(cv.resize(low_disparity, dim), low_params["min_disparity"], low_params["disparity"])
    
    return high, mid, low
