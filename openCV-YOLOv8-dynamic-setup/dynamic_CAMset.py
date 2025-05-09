import cv2
import numpy as np
from ultralytics import YOLO
import time
import threading
from collections import deque

def adjust_image(image, brightness=0, contrast=1.0, saturation=1.0):
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
    adjusted = np.clip(image.astype(np.int16) + brightness, 0, 255).astype(np.uint8)
    
    # Contrast adjustment (multiply by factor)
    adjusted = np.clip(adjusted.astype(np.float32) * contrast, 0, 255).astype(np.uint8)
    
    # Saturation adjustment (convert to HSV and adjust S channel)
    hsv = cv2.cvtColor(adjusted, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    adjusted = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    return adjusted

class ImageOptimizer:
    def __init__(self, model, min_conf_threshold=0.15):
        self.model = model
        self.min_conf_threshold = min_conf_threshold
        self.best_params = {"brightness": 0, "contrast": 1.0, "saturation": 1.0}
        self.best_confidence = 0.0
        self.is_optimizing = False
        self.optimization_thread = None
        self.recent_confidences = deque(maxlen=10)  # Store recent confidence values
        self.param_history = deque(maxlen=20)  # Store recent parameter combinations
        
    def evaluate_params(self, frame, brightness, contrast, saturation):
        """Evaluate a parameter combination and return the max confidence for person detection"""
        adjusted = adjust_image(frame.copy(), brightness, contrast, saturation)
        results = self.model(adjusted, conf=self.min_conf_threshold, classes=[0])
        
        if len(results[0].boxes) == 0:
            return 0.0
        
        # Get the maximum confidence score for person detections
        confidences = [box.conf.item() for box in results[0].boxes]
        return max(confidences) if confidences else 0.0
    
    def optimize_params(self, frame):
        """Run optimization to find the best parameters"""
        # Define parameter ranges to search
        brightness_range = [-50, -25, 0, 25, 50]
        contrast_range = [0.8, 0.9, 1.0, 1.1, 1.2]
        saturation_range = [0.8, 0.9, 1.0, 1.1, 1.2]
        
        best_confidence = 0.0
        best_params = self.best_params.copy()
        
        # Check if we've already tried similar parameters recently
        param_set = set((b, c, s) for b, c, s, _ in self.param_history)
        
        # Grid search through parameter combinations
        for brightness in brightness_range:
            for contrast in contrast_range:
                for saturation in saturation_range:
                    # Skip if we've tried this combination recently
                    if (brightness, contrast, saturation) in param_set:
                        continue
                        
                    confidence = self.evaluate_params(frame, brightness, contrast, saturation)
                    
                    # Store this parameter combination and its result
                    self.param_history.append((brightness, contrast, saturation, confidence))
                    
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_params = {
                            "brightness": brightness,
                            "contrast": contrast,
                            "saturation": saturation
                        }
        
        # Update best parameters if we found better ones
        if best_confidence > self.best_confidence:
            self.best_confidence = best_confidence
            self.best_params = best_params
            print(f"New best parameters found: {self.best_params}, confidence: {self.best_confidence:.2f}")
        
        self.is_optimizing = False
    
    def start_optimization(self, frame):
        """Start the optimization process in a separate thread"""
        if not self.is_optimizing:
            self.is_optimizing = True
            self.optimization_thread = threading.Thread(target=self.optimize_params, args=(frame.copy(),))
            self.optimization_thread.daemon = True
            self.optimization_thread.start()
    
    def update_confidence(self, confidence):
        """Update the recent confidence values"""
        self.recent_confidences.append(confidence)
        
        # If confidence is dropping, trigger optimization
        if len(self.recent_confidences) >= 5:
            avg_recent = sum(list(self.recent_confidences)[-5:]) / 5
            if avg_recent < self.best_confidence * 0.8 and not self.is_optimizing:
                return True  # Signal to start optimization
        return False
    
    def get_best_params(self):
        """Get the current best parameters"""
        return self.best_params

def run_yolo_detection():
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')  # Use the nano model, you can also use 's', 'm', 'l', or 'x' for different sizes
    
    # Open the video capture (0 for webcam, or provide a video file path)
    cap = cv2.VideoCapture(0)
    
    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Get the video frame dimensions
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Set up FPS calculation
    prev_time = 0
    new_time = 0
    
    # Create window and trackbars
    cv2.namedWindow("Person Detection")
    cv2.createTrackbar("Brightness", "Person Detection", 100, 200, lambda x: None)  # 0-200 (100 is neutral)
    cv2.createTrackbar("Contrast", "Person Detection", 100, 300, lambda x: None)    # 0-300 (100 is neutral)
    cv2.createTrackbar("Saturation", "Person Detection", 100, 300, lambda x: None)  # 0-300 (100 is neutral)
    cv2.createTrackbar("Confidence", "Person Detection", 25, 100, lambda x: None)   # 0-100 (25 = 0.25)
    cv2.createTrackbar("Auto Optimize", "Person Detection", 0, 1, lambda x: None)   # 0=off, 1=on
    
    # Initialize the image optimizer
    optimizer = ImageOptimizer(model, min_conf_threshold=0.25)
    
    # Flag to track if we're in auto or manual mode
    auto_mode = False
    last_optimization_time = 0
    optimization_interval = 2  # seconds between optimizations
    
    while True:
        # Read a frame from the video
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image")
            break
        
        # Check if auto mode is enabled
        auto_mode = cv2.getTrackbarPos("Auto Optimize", "Person Detection") == 1
        
        if auto_mode:
            # Use the best parameters from the optimizer
            params = optimizer.get_best_params()
            brightness_val = params["brightness"]
            contrast_val = params["contrast"]
            saturation_val = params["saturation"]
            
            # Update trackbars to reflect auto values (add 100 to brightness to fit 0-200 range)
            cv2.setTrackbarPos("Brightness", "Person Detection", int(brightness_val + 100))
            cv2.setTrackbarPos("Contrast", "Person Detection", int(contrast_val * 100))
            cv2.setTrackbarPos("Saturation", "Person Detection", int(saturation_val * 100))
        else:
            # Get current trackbar values
            brightness_val = cv2.getTrackbarPos("Brightness", "Person Detection") - 100  # -100 to 100
            contrast_val = cv2.getTrackbarPos("Contrast", "Person Detection") / 100.0    # 0.0 to 3.0
            saturation_val = cv2.getTrackbarPos("Saturation", "Person Detection") / 100.0  # 0.0 to 3.0
        
        # Get confidence threshold
        conf_threshold = cv2.getTrackbarPos("Confidence", "Person Detection") / 100.0  # 0.0 to 1.0
        
        # Apply adjustments to the frame
        adjusted_frame = adjust_image(
            frame, 
            brightness=brightness_val, 
            contrast=contrast_val, 
            saturation=saturation_val
        )
        
        # Calculate FPS
        new_time = time.time()
        fps = 1 / (new_time - prev_time) if (new_time - prev_time) > 0 else 0
        prev_time = new_time
        
        # Filter for person class only (class 0)
        results = model(adjusted_frame, conf=conf_threshold, classes=[0])
        annotated_frame = results[0].plot()
        
        # Count persons detected
        person_count = len(results[0].boxes)
        
        # Get max confidence if persons detected
        current_max_conf = 0
        if person_count > 0:
            confidences = [box.conf.item() for box in results[0].boxes]
            current_max_conf = max(confidences)
            
            # Update optimizer with current confidence
            if auto_mode:
                should_optimize = optimizer.update_confidence(current_max_conf)
                
                # Start optimization if needed and not already optimizing
                current_time = time.time()
                if should_optimize and not optimizer.is_optimizing and (current_time - last_optimization_time) > optimization_interval:
                    print("Starting parameter optimization...")
                    optimizer.start_optimization(frame)
                    last_optimization_time = current_time
        
        # Add information to the frame
        cv2.putText(annotated_frame, f'FPS: {int(fps)}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Brightness: {brightness_val}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Contrast: {contrast_val:.1f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Saturation: {saturation_val:.1f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Confidence: {conf_threshold:.2f}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Persons detected: {person_count}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Max confidence: {current_max_conf:.2f}', (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_frame, f'Mode: {"Auto" if auto_mode else "Manual"}', (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        if optimizer.is_optimizing:
            cv2.putText(annotated_frame, "OPTIMIZING...", (frame_width - 200, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Display the annotated frame
        cv2.imshow("Person Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def detect_on_image(image_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Create window and trackbars
    cv2.namedWindow("Person Detection")
    cv2.createTrackbar("Brightness", "Person Detection", 100, 200, lambda x: None)  # 0-200 (100 is neutral)
    cv2.createTrackbar("Contrast", "Person Detection", 100, 300, lambda x: None)    # 0-300 (100 is neutral)
    cv2.createTrackbar("Saturation", "Person Detection", 100, 300, lambda x: None)  # 0-300 (100 is neutral)
    cv2.createTrackbar("Confidence", "Person Detection", 25, 100, lambda x: None)   # 0-100 (25 = 0.25)
    cv2.createTrackbar("Auto Optimize", "Person Detection", 0, 1, lambda x: None)   # 0=off, 1=on
    
    # Initialize the image optimizer
    optimizer = ImageOptimizer(model, min_conf_threshold=0.25)
    
    # Run optimization once at the start for the image
    print("Running initial parameter optimization...")
    optimizer.optimize_params(img)
    
    while True:
        # Check if auto mode is enabled
        auto_mode = cv2.getTrackbarPos("Auto Optimize", "Person Detection") == 1
        
        if auto_mode:
            # Use the best parameters from the optimizer
            params = optimizer.get_best_params()
            brightness_val = params["brightness"]
            contrast_val = params["contrast"]
            saturation_val = params["saturation"]
                        # Update trackbars to reflect auto values (add 100 to brightness to fit 0-200 range)
            cv2.setTrackbarPos("Brightness", "Person Detection", int(brightness_val + 100))
            cv2.setTrackbarPos("Contrast", "Person Detection", int(contrast_val * 100))
            cv2.setTrackbarPos("Saturation", "Person Detection", int(saturation_val * 100))
        else:
            # Get current trackbar values
            brightness_val = cv2.getTrackbarPos("Brightness", "Person Detection") - 100  # -100 to 100
            contrast_val = cv2.getTrackbarPos("Contrast", "Person Detection") / 100.0    # 0.0 to 3.0
            saturation_val = cv2.getTrackbarPos("Saturation", "Person Detection") / 100.0  # 0.0 to 3.0
        
        # Get confidence threshold
        conf_threshold = cv2.getTrackbarPos("Confidence", "Person Detection") / 100.0  # 0.0 to 1.0
        
        # Apply adjustments to the image
        adjusted_img = adjust_image(
            img.copy(), 
            brightness=brightness_val, 
            contrast=contrast_val, 
            saturation=saturation_val
        )
        
        # Filter for person class only (class 0)
        results = model(adjusted_img, conf=conf_threshold, classes=[0])
        annotated_img = results[0].plot()
        
        # Count persons detected
        person_count = len(results[0].boxes)
        
        # Get max confidence if persons detected
        current_max_conf = 0
        if person_count > 0:
            confidences = [box.conf.item() for box in results[0].boxes]
            current_max_conf = max(confidences)
        
        # Add information to the image
        cv2.putText(annotated_img, f'Brightness: {brightness_val}', (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Contrast: {contrast_val:.1f}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Saturation: {saturation_val:.1f}', (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Confidence: {conf_threshold:.2f}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Persons detected: {person_count}', (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Max confidence: {current_max_conf:.2f}', (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(annotated_img, f'Mode: {"Auto" if auto_mode else "Manual"}', (20, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the annotated image
        cv2.imshow("Person Detection", annotated_img)
        
        # Break the loop if 'q' is pressed, save if 's' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save the current adjusted and annotated image
            output_path = f"person_detection_{int(time.time())}.jpg"
            cv2.imwrite(output_path, annotated_img)
            print(f"Saved image to {output_path}")
        elif key == ord('o'):
            # Manually trigger optimization
            print("Running parameter optimization...")
            optimizer.optimize_params(img)
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Choose whether to run on webcam/video or on a single image
    use_webcam = False
    
    if use_webcam:
        run_yolo_detection()
    else:
        # Provide the path to your image
        image_path ="./img2.png"
        detect_on_image(image_path)
