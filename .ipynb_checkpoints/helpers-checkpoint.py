def normalize(disparity, num_dispar):
    """
    Normalize the disparity for equal treatment
    """
    scale_disparity = disparity / num_dispar
    min_value = scale_disparity.min()
    max_value = scale_disparity.max()
    
    return (scale_disparity - min_value) / (max_value - min_value)

def depth_helper(imgL, imgR, params):
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
        # P1 = 8 * 3 * kernel ** 2,
    #     P2 = 32 * 3 * kernel ** 2,
        # disp12MaxDiff = 5,
        # uniquenessRatio = 2,
    #     speckleWindowSize = 10,
    #     speckleRange = 32
    )

    # Return disparity
    return stereo.compute(blur_imgL, blur_imgR)

def calculate_depth_dim(imgL, imgR, params):
    """
    Calculate the disparity given a specific dimension and parameters
    """
    # Deep copy
    imgL, imgR = imgL.copy(), imgR.copy()
    
    # Resize
    dim = params["dimension"]
    resized_imgL, resized_imgR = cv.resize(imgL, dim), cv.resize(imgR, dim) # Resize
    
    # Calculate disparity and return
    return depth_helper(resized_imgL, resized_imgR, params)

def combine_disparity(dim, low, mid, high):
    combined = low.copy()
    
    for i in range(dim[1]):
        for j in range(dim[0]): 
            low_value, mid_value, high_value = low[i][j], mid[i][j], high[i][j]
            if low_value < mid_value or low_value < high_value:
                combined[i][j] = (low_value * 3 + mid_value * 1 + high_value) / 5
    
    return combined

def return_depth(imgL, imgR):
    """
    Calculate the disparity of the input images
    """
    # Original dimension
    dim = imgL.shape[::-1]
    
    # 720p
    hi_params = {"dimension": (1280, 720), "kernel": 3, "block": 15,
              "min_disparity": 5, "disparity": 10}
    hi_disparity = calculate_depth_dim(imgL, imgR, hi_params)
    
    # 360p
    mid_params = {"dimension": (640, 360), "kernel": 7, "block": 18,
              "min_disparity": 0, "disparity": 8}
    mid_disparity = calculate_depth_dim(imgL, imgR, mid_params)
    
    # 180p
    low_params = {"dimension": (320, 180), "kernel": 11, "block": 20,
              "min_disparity": 0, "disparity": 5}
    low_disparity = calculate_depth_dim(imgL, imgR, low_params)
    
    # Resize and normalize
    high = normalize(cv.resize(hi_disparity, dim), hi_params["disparity"])
    mid = normalize(cv.resize(mid_disparity, dim), mid_params["disparity"])
    low = normalize(cv.resize(low_disparity, dim), low_params["disparity"])

    # Combine
    # disparity = np.maximum.reduce([low, mid, high])
    disparity = combine_disparity(dim, low, mid, high)
    
    return disparity, high, mid, low