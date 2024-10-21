import cv2
import numpy as np
import dlib
from imutils import face_utils

# Load dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

def detect_pupil_dilation(eye_region):
    # Convert the eye region to grayscale
    gray_eye = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to enhance pupil detection
    _, thresholded_eye = cv2.threshold(gray_eye, 30, 255, cv2.THRESH_BINARY_INV)

    # Find contours to detect the pupil
    contours, _ = cv2.findContours(thresholded_eye, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If contours (pupil detection) found
    if contours:
        # Get the largest contour, corresponding to the pupil
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the bounding circle for the pupil
        ((cx, cy), radius) = cv2.minEnclosingCircle(largest_contour)

        # Return the radius and center coordinates of the pupil
        return radius, (int(cx), int(cy))
    return None, None

def analyze_pupil_tiredness(frame, face, shape):
    # Extract left and right eyes from the face landmarks
    left_eye_pts = shape[36:42]  # Left eye landmarks
    right_eye_pts = shape[42:48]  # Right eye landmarks

    # Analyze both eyes for pupil dilation
    for eye_pts in [left_eye_pts, right_eye_pts]:
        (x, y, w, h) = cv2.boundingRect(np.array([eye_pts]))
        eye_region = frame[y:y+h, x:x+w]

        # Detect pupil dilation in the eye region
        radius, center = detect_pupil_dilation(eye_region)

        # Mark the pupil on the original frame if it is dilated
        if radius and radius > 15:  # Adjust this radius for dilation threshold
            cv2.circle(frame, (center[0] + x, center[1] + y), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, "Dilated", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            return True
    return False

def main():
    cap = cv2.VideoCapture(1)  # Use camera 0 for default laptop camera

    if not cap.isOpened():
        print("Error: Unable to access camera")
        return
    
    tiredness = 0  # Counter for tiredness detection

    while True:
        ret, frame = cap.read(0)
        if not ret:
            print("Error: Unable to capture frame")
            break

        # Convert the frame to grayscale for dlib face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using dlib
        faces_dlib = detector(gray, 0)

        for face_rect in faces_dlib:
            # Get facial landmarks
            shape = predictor(gray, face_rect)
            shape = face_utils.shape_to_np(shape)

            # Analyze pupil dilation and tiredness
            if analyze_pupil_tiredness(frame, face_rect, shape):
                tiredness += 1  # Increment if pupil dilation is detected
            else:
                tiredness = max(0, tiredness - 1)  # Decrease count when no dilation

            # Detect tiredness if prolonged pupil dilation is detected
            if tiredness > 30:  # Tweak the threshold based on application
                cv2.putText(frame, "Tired", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        # Show the frame with annotations
        cv2.imshow("Pupil and Tiredness Detection", frame)

        # Press 'q' to exit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
