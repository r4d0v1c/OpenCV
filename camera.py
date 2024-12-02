import cv2

#Open default camera, 0 is the first camera, 1 would be the second etc.
cam = cv2.VideoCapture(0)

#Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))


#Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    #ret -> if the frame is successfully captured or not
    ret, frame = cam.read()

    #Write the frame ot the output file
    out.write(frame)

    #Display the captured frame
    cv2.imshow('Camera', frame)

    #Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

#Release the capture and writing objects
cam.release()
out.release()
cv2.destroyAllWindows()