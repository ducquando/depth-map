import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.helpers import return_ranges, return_depth, return_disparity, normalize, combine_disparity, merge_range, display_individual_depth, draw_text, px_cm_ratio

def display_depth(imgL, imgR, stereo):
    # Convert to grayscale
    gray_imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    gray_imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)

    # Calculate disparity
    hi, mid, low = return_disparity(gray_imgL, gray_imgR)

    # Specify the dectected range to be 30cm -> 2m
    range_low = return_ranges(low, 100 * px_cm_ratio(return_depth(200), False),
                              100 * px_cm_ratio(return_depth(30), False), 5, 100000)

    # Specify the dectected range to be 2m -> 25m
    range_mid_high = merge_range(return_ranges(low, 100 * px_cm_ratio(return_depth(2500), False),
                                               100 * px_cm_ratio(return_depth(200), False), 10, 50000))

    # Combine disparity of different resolution
    combined = combine_disparity(gray_imgL.shape[::-1], low, mid, hi, range_mid_high)

    # Show depth and warning text
    canvas_dispar = normalize(combined)
    canvas_img = imgL.copy()
    for dispar_range in range_low:
        x_left, y_top, x_right, y_bot, value = display_individual_depth(combined, dispar_range, stereo)

        # Put text on disparity image
        draw_text(canvas_dispar, "Object at %.2f cm" % value, (x_left + 5, y_top - 10))

        # Put text and detected box on image imae
        cv.putText(canvas_img, "Object at %.2f cm" % value,
                   (x_left + 5, y_top - 10), 1, 2, (0, 255, 0), 2, 2)
        cv.rectangle(canvas_img, (x_left, y_top), (x_right, y_bot), (0, 255, 0), 2)

        # Put warning text
        if value < 50:
            draw_text(canvas_dispar, "WARNING: Object closer than 50cm!", (50, 50))
            cv.putText(canvas_img, "WARNING: Object closer than 50cm!",
                       (50, 50), 1, 2, (255, 0, 0), 2, 2)
    
    # Displaying the result
    cv.imshow("Disparity", canvas_dispar)
    cv.imshow("Real-life", canvas_img)
    

def main():
    CamL = cv.VideoCapture(0) # Camera ID for left camera
    CamR = cv.VideoCapture(1) # Camera ID for right camera
    
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
