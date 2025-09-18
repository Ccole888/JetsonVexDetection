from ultralytics import YOLO
import cv2

# Load your trained model
model = YOLO('../runs/detect/vex_cubes_detection/weights/best.pt')

CONFIDENCE_THRESHOLD = 0.70
# Open the webcam (source 0 is usually the built-in camera)
cap = cv2.VideoCapture(0)

# Loop through the webcam frames
while cap.isOpened():
    # Read a frame from the webcam
    success, frame = cap.read()
    
    if success:
        # Run YOLOv8 inference on the frame
        results = model(frame, conf=CONFIDENCE_THRESHOLD)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()  # This creates a frame with boxes and labels
        
        # Display the annotated frame
        cv2.imshow('YOLOv8 Webcam Detection', annotated_frame)
        
        # Break the loop if 'q' or 'Esc' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 is the ASCII for Esc key
            break
    else:
        # Break the loop if the frame is not read successfully
        break

# Release the webcam and close the display window
cap.release()
cv2.destroyAllWindows()