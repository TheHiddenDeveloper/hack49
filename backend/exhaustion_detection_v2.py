import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression

# Model to process whether the pupil is dialated or not
class PupilSizeModel:
    def __init__(self):
        self.model = LogisticRegression()
    
    def is_dilated(self, pupil_size):
        return pupil_size[1] > 20 # tweak this pixel counts so that the dilation is calibrated properly

# Load pre-trained model for pupil size classification
pupil_model = PupilSizeModel()

# Load pre-trained face and eye detectors using Haar Cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

"""
Description: Detect whether the eyes's pupil is dilated.
Parameter: 
    eye_region: The region of the eyes.
Return: the radius and coordiantes of the eyes.
"""
def detect_pupil_dilation(eye_region):
    # Convert the eye region to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance pupil detection
    _, thresholded_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the pupil
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Get the largest contour, corresponding to the pupil
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding circle for the pupil
        eye_radius = cv2.minEnclosingCircle(largest_contour)

        return eye_radius  # Return the radius (size) of the pupil
    return None

"""
Description: Check if a person is tired using the sizes of their pupils
Parameter:
    left_pupil_size: the size of the left pupil
    right_pupil_size: the size of the right pupil
    resizing: whether the size is changing (False by default)
Return: Feedback whether a person is tired, likely tired, or normal.
"""
def check_tiredness(left_pupil_size, right_pupil_size, resizing):
    left_dilated = pupil_model.is_dilated(left_pupil_size)
    right_dilated = pupil_model.is_dilated(right_pupil_size)
    
    if not resizing and (left_dilated or right_dilated):
        return "The person is tired as pupils are dilated and not resizing."
    elif resizing and (left_dilated or right_dilated):
        return "The person is likely tired as pupils are dilated despite resizing."
    elif resizing and not left_dilated and not right_dilated:
        return "The person appears normal with normal pupil resizing."
    return "Unable to determine tiredness."

def analyze_pupil_behavior(video_stream):
    prev_left_pupil_size = prev_right_pupil_size = None
    resizing = False

    while True:
        ret, frame = video_stream.read()
        if not ret:
            print("Error: Unable to capture video frames.")
            break

        # Convert to grayscale for faster processing
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray_frame, 1.3, 5)

        for (x, y, w, h) in faces:
            # Draw a rectangle around the face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            # Region of interest for face (where the eyes are located)
            face_region = gray_frame[y:y+h, x:x+w]
            color_face_region = frame[y:y+h, x:x+w]

            # Limit eye detection to the top half of the face (to avoid detecting the mouth or nose)
            upper_face_region = face_region[:h//2, :]
            upper_color_face_region = color_face_region[:h//2, :]

            # Detect eyes within the upper face region
            eyes = eye_cascade.detectMultiScale(upper_face_region)

            # Sort the eyes based on their size (area), and pick the two largest (most likely to be left and right eyes)
            eyes = sorted(eyes, key=lambda e: e[2] * e[3], reverse=True)[:2]

            if len(eyes) == 2:
                # Extract the two eye regions
                for (ex, ey, ew, eh) in eyes:
                    eye_region = upper_color_face_region[ey:ey+eh, ex:ex+ew]

                    # Detect pupil size
                    pupil_size = detect_pupil_dilation(eye_region)

                    if pupil_size:
                        
                        # Save the pupil size to analyze resizing
                        if prev_left_pupil_size is None:
                            prev_left_pupil_size = pupil_size
                        else:
                            resizing = abs(pupil_size[1] - prev_left_pupil_size[1]) > 1
                            prev_left_pupil_size = pupil_size

            # Assess tiredness based on pupil behavior
            if prev_left_pupil_size is not None:
                status = check_tiredness(prev_left_pupil_size, prev_left_pupil_size, resizing)
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        # Display the frame with status
        cv2.imshow("Pupil Behavior Analysis", frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video stream and close windows
    video_stream.release(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Open the webcam video stream
    video_capture = cv2.VideoCapture(0)

    if not video_capture.isOpened():
        print("Error: Unable to access webcam.")
    else:
        analyze_pupil_behavior(video_capture)
