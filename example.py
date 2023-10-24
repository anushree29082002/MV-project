import cv2
from gaze_tracking import GazeTracking

# Load the cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize GazeTracking
gaze = GazeTracking()

# Initialize a counter for consecutive frames where blinking is detected
blink_counter = 0
blink_threshold = 5  # Adjust this threshold for blink detection sensitivity

# To capture video from the webcam
webcam = cv2.VideoCapture(0)

while True:
    # Read a new frame from the webcam
    _, frame = webcam.read()

    # Send the frame to GazeTracking for analysis
    gaze.refresh(frame)
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect the faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    frame = gaze.annotated_frame()
    text = ""
    
    # Set the default rectangle color to green
    rect_color = (0, 255, 0)

    # Check for blinking
    if gaze.is_blinking():
        blink_counter += 1
        if blink_counter >= blink_threshold:
            text = "Blinking"
            # Set the rectangle color to red for blinking
            rect_color = (0, 0, 255)
    else:
        blink_counter = 0
        # Check the horizontal gaze ratio to determine left or right gaze
        if gaze.horizontal_ratio() is not None:
            if gaze.horizontal_ratio() < 0.3:
                text = "Looking left"
                # Set the rectangle color to red for looking left
                rect_color = (0, 0, 255)
            elif gaze.horizontal_ratio() > 0.7:
                text = "Looking right"
                # Set the rectangle color to red for looking right
                rect_color = (0, 0, 255)
            elif 0.3 <= gaze.horizontal_ratio() <= 0.7:
                text = "Looking center"

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()

    # Draw the rectangle with the determined color
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), rect_color, 2)

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break

webcam.release()
cv2.destroyAllWindows()
