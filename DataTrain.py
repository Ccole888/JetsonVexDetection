from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
import numpy as np
import threading

# Load your trained model
model = YOLO('../runs/detect/vex_cubes_detection/weights/best.pt')

CONFIDENCE_THRESHOLD = 0.70

# Function to display polar vector visualization
def show_polar_vector():
    r = 1
    theta = 0  # radians

    fig = plt.figure("Polar Vector Display", facecolor='black')
    ax = fig.add_subplot(111, polar=True)
    ax.set_facecolor('black')

    # Plot the vector
    ax.plot([0, theta], [0, r], color='cyan', linewidth=2)
    ax.scatter(theta, r, color='cyan', s=50)

    # Draw concentric circles at r = 1, 2, 3...
    max_r = 5
    ax.set_ylim(0, max_r)
    ax.set_yticks(range(1, max_r + 1))
    ax.set_yticklabels([str(i) for i in range(1, max_r + 1)], color='white')
    ax.set_xticklabels(['0', 'π/4', 'π/2', '3π/4', 'π', '5π/4', '3π/2', '7π/4'], color='white')

    # Customize grid and labels
    ax.grid(color='gray', linestyle='dotted')
    ax.set_title("Polar Vector: r=1, θ=0", color='white')

    plt.show()

# Launch polar vector display in a separate thread
threading.Thread(target=show_polar_vector, daemon=True).start()

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
