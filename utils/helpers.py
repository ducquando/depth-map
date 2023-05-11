import numpy as np
import cv2 as cv

def normalize(value: np.ndarray):
    """
    Normalize the input to [0, 1] for visualization
    """
    min_value = value.min()
    max_value = value.max()
    
    return (value - min_value) / (max_value - min_value)


def merge_range(input_range):
    """
    Merge ranges together
    """
    # Cast for empty input
    if len(input_range) == 0:
        return input_range
    
    final_range = [input_range[0]]
    for i in range(1, len(input_range)):
        if final_range[-1][1] == input_range[i][0]:
            final_range[-1][1] = input_range[i][1]
        else:
            final_range.append(input_range[i])
            
    return final_range


def draw_text(img, text, pos = (0, 0), font = 1, font_scale = 2, 
              text_color = (1, 1, 1), font_thickness = 2, lineType = 2,
              text_color_bg = (0, 0, 0)):
    """
    Draw text with background color
    """
    x, y = pos
    text_size, _ = cv.getTextSize(text, font, font_scale, font_thickness)
    text_w, text_h = text_size
    cv.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness, lineType)