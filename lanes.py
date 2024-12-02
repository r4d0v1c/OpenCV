import cv2
import numpy as np
#import matplotlib.pyplot as plt

#0   - > black
#255 - > white

#x -> image_width (columns) r*c = total_pixels_of_img
#y -> image_height (rows)

#Gradient: Measure of change in brightness over adjacent pixels (derivative)
#gray-scale image 1ch/px 
#rgb-scale image 3ch/px

#canny fnc internally applies Gaussian 5x5 when it's called, computes the gradient in all directions of blured image

def make_coordinates(image, line_parameters):
    slope, intercept = line_parameters
    print(image.shape)#height, width, number of channels (704 bottom of image, max y value)
    y1 = image.shape[0] #height (704)
    y2 = int(y1*(3/5)) #422.4 -> start from the bottom and goes upwards 3/5 of y1
    #x = (y-b)/m
    x1 = int((y1 - intercept)/slope)
    x2 = int((y2 - intercept)/slope)
    return np.array([x1, y1, x2, y2])


def averaged_slope_intercept(image, lines):
    left_fit = []  # Stores left line slopes and intercepts
    right_fit = []  # Stores right line slopes and intercepts

    if lines is None or len(lines) == 0:  # If no lines are detected
        return None

    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Degree 1 polynomial fit
        slope = parameters[0]
        intercept = parameters[1]
        if slope < 0:
            left_fit.append((slope, intercept))
        else:
            right_fit.append((slope, intercept))

    left_line = None
    right_line = None

    # Calculate averages only if the respective list is not empty
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        if isinstance(left_fit_average, np.ndarray) and len(left_fit_average) == 2:
            left_line = make_coordinates(image, left_fit_average)

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        if isinstance(right_fit_average, np.ndarray) and len(right_fit_average) == 2:
            right_line = make_coordinates(image, right_fit_average)

    if left_line is not None and right_line is not None:
        return np.array([left_line, right_line])
    elif left_line is not None:
        return np.array([left_line])
    elif right_line is not None:
        return np.array([right_line])
    else:
        return None


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) #convert to gray image
    blur = cv2.GaussianBlur(gray, (5,5), 0) #to reduce noice in gray-scale image, smoothening image, low pass filter
    canny = cv2.Canny(blur, 50, 150) #to outline the strongest gradients in image
    return canny

def region_of_interest(image):
    height = image.shape[0] #700 whole y axis

    #array of polygones
    polygones = np.array([
    [(200, height), (1100, height), (550, 250)]
    ])
    
    mask = np.zeros_like(image) #same dimensions
    cv2.fillPoly(mask, polygones, 255) #apply triangle on the mask
    masked_image = cv2.bitwise_and(image, mask) #shows only region of interest(lane_line)
    return masked_image

def display_lines(image, lines):
    line_image = np.zeros_like(image) #has the same dimensions as image (black)
    if lines is not None: #if the array is not empty
        for x1,y1,x2,y2 in lines:
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10) #last is line thick
    return line_image
'''
    gradients that exceed high treshold are traced as bright pixels identifying adjacent pixels in the image with the most rapid changes in the brightness
    small changes in brigtness are not chased at all and accordingly they are black bcs they're bellow lower treshold'''
'''
   if the gradient is larger than upper treshold then it is accepted as an edge pixel
   if it is below the lower treshold it is rejected
   if it is between these two it will be accepted only if it is connected to a strong edge
   recommended ratio (1-2) or (1-3)
'''

#image = cv2.imread('test_image.jpg')
#lane_image = np.copy(frame) #so not to affect actual image

cap = cv2.VideoCapture("test2.mp4")
while(cap.isOpened()):
    _, frame = cap.read() #return smth bool and frame
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5) # 2px(size of bin), np.pi/180 = 1 degree
    #treshold : minimum number of votes nedded to accept a candidate line(4th param.)
    #mineLineLength : any detected lines traced by less than 40 pixels are rejected
    #maxLineGap : maximum distance in pixels between segmented lines which we will allow to be connected in single line


    averaged_lines = averaged_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines) #purple lines
    image_blender = cv2.addWeighted(frame, 0.8, line_image, 1, 1) #multiply all lane_image elements by 0.8
                                                                    #multiply all line_image elements by 1 has 20% more, it will be more clearly defined(brighter than lane_image)

    #cv2.imshow("smoother lines", iline_image)
    cv2.imshow("result", image_blender)

    #for how many millis a picture will be displayed, 1ms/frame
    if cv2.waitKey(1) == ord('q'): 
        break
cap.release()
cap.destroyAllWindows()