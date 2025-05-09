import cv2  # Import OpenCV library for image processing
import numpy as np  # Import NumPy for numerical operations
from ultralytics import YOLO  # Import YOLO model from ultralytics
import time  # Import time module for FPS calculation

def adjust_image(image, brightness=0, contrast=1.0, saturation=1.0):  # Define function to adjust image parameters
    """
    Adjust brightness, contrast, and saturation of an image.
    
    Args:
        image: Input BGR image
        brightness: Brightness adjustment (-100 to 100)
        contrast: Contrast adjustment (0.0 to 3.0)
        saturation: Saturation adjustment (0.0 to 3.0)
    
    Returns:
        Adjusted image
    """
    # Brightness adjustment (add/subtract value)
    adjusted = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)  # Convert image to 16-bit integer, add brightness, clip values, convert back to 8-bit
    
    # Contrast adjustment (multiply by factor)
    adjusted = np.clip(adjusted.astype(np.float32) * contrast, 0, 255).astype(np.uint8)  # Convert to float, multiply by contrast factor, clip values, convert back to 8-bit
    
    # Saturation adjustment (convert to HSV and adjust S channel)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)  # Convert image to HSV color space
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)  # Adjust saturation channel
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)  # Convert back to BGR color space
    
    return adjusted  # Return the adjusted image

def run_yolo_detection():  # Define function for real-time YOLO detection
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Initialize YOLO model with nano weights
    
    # Open the video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(0)  # Create video capture object for webcam
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():  # Check if camera opened successfully
        print("Error: Could not open video source.")  # Print error message
        return  # Exit function if camera not opened
    
    # Get the video frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Get frame width from camera
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Get frame height from camera
    
    # Set up FPS calculation
    prev_time = 0  # Initialize previous time for FPS calculation
    new_time = 0  # Initialize new time for FPS calculation
    
    # Initial adjustment values
    brightness = 0  # Initialize brightness value
    contrast = 1.0  # Initialize contrast value
    saturation = 1.0  # Initialize saturation value
    
    # Create window and trackbars
    cv2.namedWindow("YOLOv8 Detection")  # Create named window for display
    cv2.createTrackbar("Brightness", "YOLOv8 Detection", 100, 200, lambda x: None)  # Create brightness trackbar
    cv2.createTrackbar("Contrast", "YOLOv8 Detection", 100, 300, lambda x: None)    # Create contrast trackbar
    cv2.createTrackbar("Saturation", "YOLOv8 Detection", 100, 300, lambda x: None)  # Create saturation trackbar
    
    while True:  # Main processing loop
        # Read a frame from the video
        ret, frame = cap.read()  # Capture frame from camera
        if not ret:  # Check if frame was captured successfully
            print("Error: Failed to capture image")  # Print error message
            break  # Exit loop if frame capture failed
        
        # Get current trackbar values
        brightness_val = cv2.getTrackbarPos("Brightness", "YOLOv8 Detection") - 100  # Get brightness value from trackbar
        contrast_val = cv2.getTrackbarPos("Contrast", "YOLOv8 Detection") / 100.0    # Get contrast value from trackbar
        saturation_val = cv2.getTrackbarPos("Saturation", "YOLOv8 Detection") / 100.0  # Get saturation value from trackbar
        
        # Apply adjustments to the frame
        adjusted_frame = adjust_image(  # Apply image adjustments using the adjust_image function
            frame, 
            brightness=brightness_val, 
            contrast=contrast_val, 
            saturation=saturation_val
        )
        
        # Calculate FPS
        new_time = time.time()  # Get current time
        fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0  # Calculate FPS
        prev_time = new_time  # Update previous time
        
        # Run YOLOv8 inference on the adjusted frame
        results = model(adjusted_frame)  # Perform object detection
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()  # Draw detection results on frame
        
        # Add FPS information and adjustment values
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add FPS text
        cv2.putText(annotated_frame, f'Brightness: {brightness_val}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add brightness value
        cv2.putText(annotated_frame, f'Contrast: {contrast_val:.1f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add contrast value
        cv2.putText(annotated_frame, f'Saturation: {saturation_val:.1f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add saturation value
        
        # Display the annotated frame
        cv2.imshow("YOLOv8 Detection", annotated_frame)  # Show the processed frame
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Check for 'q' key press
            break  # Exit loop if 'q' pressed
    
    # Release the video capture object and close all windows
    cap.release()  # Release camera
    cv2.destroyAllWindows()  # Close all OpenCV windows

def detect_on_image(image_path):  # Define function for single image detection
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Initialize YOLO model
    
    # Read the image
    img = cv2.imread(image_path)  # Load image from file
    if img is None:  # Check if image was loaded successfully
        print(f"Error: Could not read image from {image_path}")  # Print error message
        return  # Exit function if image not loaded
    
    # Create window and trackbars
    cv2.namedWindow("YOLOv8 Detection")  # Create named window
    cv2.createTrackbar("Brightness", "YOLOv8 Detection", 100, 200, lambda x: None)  # Create brightness trackbar
    cv2.createTrackbar("Contrast", "YOLOv8 Detection", 100, 300, lambda x: None)    # Create contrast trackbar
    cv2.createTrackbar("Saturation", "YOLOv8 Detection", 100, 300, lambda x: None)  # Create saturation trackbar
    
    while True:  # Main processing loop
        # Get current trackbar values
        brightness_val = cv2.getTrackbarPos("Brightness", "YOLOv8 Detection") - 100  # Get brightness value
        contrast_val = cv2.getTrackbarPos("Contrast", "YOLOv8 Detection") / 100.0    # Get contrast value
        saturation_val = cv2.getTrackbarPos("Saturation", "YOLOv8 Detection") / 100.0  # Get saturation value
        
        # Apply adjustments to the image
        adjusted_img = adjust_image(  # Apply image adjustments
            img.copy(), 
            brightness=brightness_val, 
            contrast=contrast_val, 
            saturation=saturation_val
        )
        
        # Run YOLOv8 inference on the adjusted image
        results = model(adjusted_img)  # Perform object detection
        
        # Visualize the results on the image
        annotated_img = results[0].plot()  # Draw detection results
        
        # Add adjustment values
        cv2.putText(annotated_img, f'Brightness: {brightness_val}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add brightness value
        cv2.putText(annotated_img, f'Contrast: {contrast_val:.1f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add contrast value
        cv2.putText(annotated_img, f'Saturation: {saturation_val:.1f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  # Add saturation value
        
        # Display the annotated image
        cv2.imshow("YOLOv8 Detection", annotated_img)  # Show processed image
        
        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF  # Get keyboard input
        if key == ord('q'):  # Check for 'q' key
            break  # Exit loop if 'q' pressed
        elif key == ord('s'):  # Check for 's' key
            # Save the current adjusted and annotated image
            output_path = f"output_{int(time.time())}.jpg"  # Generate output filename
            cv2.imwrite(output_path, annotated_img)  # Save image to file
            print(f"Saved image to {output_path}")  # Print save confirmation
    
    cv2.destroyAllWindows()  # Close all OpenCV windows

if __name__ == "__main__":  # Main entry point of the script
    # Choose whether to run on webcam/video or on a single image
    use_webcam = True  # Flag to control webcam or image mode
    
    if use_webcam:  # Check if using webcam
        run_yolo_detection()  # Run webcam detection
    else:  # If not using webcam
        # Provide the path to your image
        image_path = "path/to/your/image.jpg"  # Set image path
        detect_on_image(image_path)  # Run image detection