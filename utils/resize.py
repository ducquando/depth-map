import cv2 as cv

def main(path, name):
    img = cv.imread(path)
    h, w = img.shape[:2]
    
    width = 360
    height = int(h / w * width)
    
    new_img = cv.resize(img, (width, height))
    cv.imwrite(name, new_img)

if __name__ == "__main__":
    main("../assets/rectify/book.png", "../assets/rectify/book.jpg")
    main("../assets/rectify/book_background.png", "../assets/rectify/book_background.jpg")