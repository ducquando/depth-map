import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

from utils.helpers import merge_range, draw_text, normalize
from utils.disparity import return_disparity, combine_disparity, return_ranges
from utils.depth import display_individual_depth

def display_depth(imgL, imgR, stereo):
    # Convert to grayscale
    gray_imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    gray_imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Calculate disparity
    hi, mid, low = return_disparity(gray_imgL, gray_imgR)

    # Specify the dectected ranges
    range_low = return_ranges(low, 0.25, 1.25, 10, 50000)
    range_mid_high = merge_range(return_ranges(low, 0.05, 0.25, 2, 50000))

    # Combine
    combined = combine_disparity(gray_imgL.shape[::-1], low, mid, hi, range_mid_high)
    
    # Create canvas for visualization
    canvas_dispar = normalize(combined)
    canvas_img = imgL.copy()

    for dispar_range in range_low:
        x_left, y_top, x_right, y_bot, value = display_individual_depth(combined, dispar_range, stereo)

        # Put text on disparity image
        draw_text(canvas_dispar, "Object at %.2f cm" % value, (x_left + 5, y_top - 10), text_color = (1, 1, 1))

        # Put text and detected box on image
        cv.putText(canvas_img, "Object at %.2f cm" % value,
                   (x_left + 5, y_top - 10), 1, 2, (0, 255, 0), 2, 2)
        cv.rectangle(canvas_img, (x_left, y_top), (x_right, y_bot), (0, 255, 0), 2)

        # Put warning text
        if value < 50:
            draw_text(canvas_dispar, "WARNING: Object closer than 50cm!", (50, 50), text_color = (1, 1, 1))
            cv.putText(canvas_img, "WARNING: Object closer than 50cm!",
                       (50, 50), 1, 2, (0, 0, 255), 2, 2)
    
    # Displaying the result
    cv.imshow("Disparity", canvas_dispar)
    cv.imshow("Real-life", canvas_img)
    

def main():
    CamL = cv.VideoCapture(2) # Camera ID for left camera
    CamR = cv.VideoCapture(0) # Camera ID for right camera
    
    while True:
        # Capturing and storing left and right camera images
        retL, imgL = CamL.read()
        retR, imgR = CamR.read()
        
        # Resize to 1280x720
        fixed_dim = (1280, 720)
        imgL, imgR = cv.resize(imgL, fixed_dim), cv.resize(imgR, fixed_dim)
        
        # Proceed only if the frames have been capture
        if retL and retR:
            display_depth(imgL, imgR, stereo = True)
 
        # Wait for 10ms before displaying the next frame
        key = cv.waitKey(10)
        if key == ord('q'):
            break
    else:
        CamL = cv.VideoCapture(2)
        CamR = cv.VideoCapture(0)
            
if __name__ == "__main__":
    main()
