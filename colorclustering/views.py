from django.shortcuts import render
import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.cluster import KMeans

def detect_face(frame, model):
    # Perform face detection using YOLOv8n-face
    results = model.predict(frame, conf=0.5)

    # Process each detected face
    for box in results[0].boxes:
        (x1, y1, x2, y2) = box.xyxy[0]

        # Draw rectangle around detected face
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

        # Extract face region
        face_region = frame[int(y1):int(y2), int(x1):int(x2)]

        # Perform skin detection
        skin_image = detect_skin(face_region)

        # Perform color clustering on the face region
        cluster_colors(face_region)

        # Evaluate face position
        face_center_x = (x1 + x2) / 2
        image_center_x = frame.shape[1] / 2
        position_good = abs(face_center_x - image_center_x) < frame.shape[1] * 0.1  # Adjust the threshold as needed

        # Evaluate lighting (example: calculate average brightness)
        avg_brightness = np.mean(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))

        # Measure face distance from the camera (example: use width of the face bounding box)
        face_distance = estimate_face_distance(x2 - x1)

        # Display evaluation results
        position_text = "Good" if position_good else "Bad"
        brightness_text = f'Avg. Brightness: {avg_brightness:.2f}'
        distance_text = f'Face Distance: {face_distance:.2f} meters'

        cv2.putText(frame, f'Face Position: {position_text}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, brightness_text, (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(frame, distance_text, (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the skin detection result
        cv2.imshow('Detected Skin', skin_image)

def detect_skin(face_region):
    # Convert the face region to the HSV color space
    hsv = cv2.cvtColor(face_region, cv2.COLOR_BGR2HSV)

    # Define a range for skin color in HSV
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Create a mask to extract skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Bitwise-AND the original image and the mask
    result = cv2.bitwise_and(face_region, face_region, mask=mask)

    return result

def cluster_colors(face_region, k=3):
    # Reshape the face region to a list of pixels
    pixels = face_region.reshape((-1, 3))

    # Convert pixel values to float
    pixels = np.float32(pixels)

    # Perform k-means clustering
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(pixels)

    # Get cluster centers and labels
    centers = np.uint8(kmeans.cluster_centers_)
    labels = kmeans.labels_

    # Map the labels to the centers
    segmented_face = centers[labels.flatten()]

    # Reshape the segmented face
    segmented_face = segmented_face.reshape(face_region.shape)

    # Display the result
    cv2.imshow('Color Clustering', segmented_face)

def estimate_face_distance(face_width):
    # Assuming standard face width of 14 centimeters at 1 meter distance
    standard_face_width = 14  # in centimeters
    standard_distance = 100  # in centimeters (1 meter)

    # Calculate the estimated face distance using the ratio of face_width to standard_face_width
    estimated_distance = (standard_distance * standard_face_width) / face_width
    return estimated_distance / 100  # convert to meters

def index(request):
    # Load the YOLOv8n-face model
    model = YOLO("yolov8n-face.pt")

    # Capture a frame from the video
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Call the function for face detection, skin detection, and color clustering
        detect_face(frame, model)

        # Display the processed frame
        cv2.imshow('Real-Time Detection', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the video capture and close
    cap.release()
    cv2.destroyAllWindows()

    # return render(request, 'colorclustering/index.html')  # Simply render the template without passing any context







