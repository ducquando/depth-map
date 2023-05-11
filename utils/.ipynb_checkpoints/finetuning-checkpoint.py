import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import re
from helpers import merge_range
from disparity import return_disparity, combine_disparity, return_ranges
from rectify import rectify, undistort
from depth import display_individual_depth

def main(output, imgL, imgR, name, stereo):
    # Convert to grayscale
    gray_imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    gray_imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    # Rectify
    retification = rectify(imgL = gray_imgL, imgR = gray_imgR, path = "../")
    rec_imgL, rec_imgR = undistort(gray_imgL, gray_imgR, retification)
    gray_imgL, gray_imgR = (rec_imgL, rec_imgR) if not stereo else (gray_imgL, gray_imgR)

    # Calculate disparity
    hi, mid, low = return_disparity(gray_imgL, gray_imgR)

    # Specify the dectected ranges
    range_low = return_ranges(low, 0.25, 1.25, 10, 50000)
    range_mid_high = merge_range(return_ranges(low, 0.05, 0.25, 2, 50000))

    # Combine disparity of different resolution
    combined = combine_disparity(gray_imgL.shape[::-1], low, mid, hi, range_mid_high)

    # Calculate depth (in pixel)
    values = []
    for dispar_range in range_low:
        _, _, _, _, value = display_individual_depth(combined, dispar_range, stereo, debug = True)
        values.append(value)
       
    # Write output depth values
    output.write(name + " : " + str(values) + "\n")
            
if __name__ == "__main__":
    # Create output file to store calib data
    output = open("../outputs/finetuning.txt", "w")
    
    # Specify the inputs
    images = glob.glob("../assets/finetuning/*left.jpg")
    
    # Read all images
    for fname in images:
        imgL = cv.imread(fname)
        imgR = cv.imread(fname.replace("left", "right"))
        stereo = True if "stereo" in fname else False
        main(output, imgL, imgR, name = (fname.replace(".jpq", "")).replace("../assets/finetuning/", ""), stereo = stereo)
    
    output.close()