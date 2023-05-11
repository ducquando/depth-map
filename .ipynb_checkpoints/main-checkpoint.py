import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from utils.depth import return_depth, px_cm_ratio, depth_to_ply, display_individual_depth
from utils.disparity import return_disparity, combine_disparity, return_ranges
from utils.rectify import rectify, undistort
from utils.helpers import normalize, merge_range, draw_text

def main(name, stereo):
    # Convert filename to paths and read images
    filepath = name.replace("_", "/")
    imgL = cv.imread('assets/' + filepath + '.jpg')
    imgR = cv.imread('assets/' + filepath.replace("left", "right") + '.jpg')
    
    # Convert to grayscale
    gray_imgL = cv.cvtColor(imgL, cv.COLOR_BGR2GRAY)
    gray_imgR = cv.cvtColor(imgR, cv.COLOR_BGR2GRAY)
    
    # Rectify
    retification = rectify(imgL = gray_imgL, imgR = gray_imgR)
    rec_imgL, rec_imgR = undistort(gray_imgL, gray_imgR, retification)
    gray_imgL, gray_imgR = (rec_imgL, rec_imgR) if not stereo else (gray_imgL, gray_imgR)

    # Calculate disparity
    hi, mid, low = return_disparity(gray_imgL, gray_imgR)

    # Specify the dectected ranges
    range_low = return_ranges(low, 0.25, 1.25, 10, 50000)
    range_mid_high = merge_range(return_ranges(low, 0.05, 0.25, 2, 50000))

    # Combine disparity of different resolution
    combined = combine_disparity(gray_imgL.shape[::-1], low, mid, hi, range_mid_high)
    
    # Calculate depth map
    depth_map = return_depth(combined, stereo)
    new_img = undistort(imgL, imgR, retification)[0] if not stereo else imgL

    # Compute point cloud
    height, width = depth_map.shape
    pcd = np.zeros((height, width, 3))

    # Remove out-of-bound area
    mask = cv.inRange(depth_map, 0, 170)
    depth_safe = cv.bitwise_and(depth_map, depth_map, mask = mask)
    norm_depth_safe = normalize(depth_safe)

    # Calculate point cloud values
    for i in range(height):
        for j in range(width):
            pcd[i][j] = [i, j, norm_depth_safe[i][j]]
        
    # Generate point cloud file from depth map
    _, data = depth_to_ply(new_img, pcd, name)
    
    # Create canvas for visualization
    canvas_dispar = normalize(combined)
    canvas_img = new_img.copy()
    
    # Show depth and warning text
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
                       (50, 50), 1, 2, (0, 0, 255), 2, 2)
            
    cv.imwrite("outputs/" + name + "_dispar.jpg", canvas_dispar) 
    cv.imwrite("outputs/" + name + ".jpg", canvas_img) 
            
if __name__ == "__main__":
     # `name`: filename + subpath -> final path: "assets/" + name + ".jpg"
    # Set `stereo` = True if using stereo camera, else False.
    main(name = "stereo_position1_left", stereo = True)
    main(name = "stereo_position2_left", stereo = True)
    main(name = "stereo_position3_left", stereo = True)
    main(name = "stereo_position4_left", stereo = True)
    main(name = "mono_position1_left", stereo = False)
    main(name = "mono_position2_left", stereo = False)