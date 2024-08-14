from ultralytics import YOLO
import cv2
import os

def main():
    # Load the trained model
    model = YOLO('yolov8/models/best_model.pt')
    
    # Load an image
    image_path = 'D:/Pg-DAI/CV/car_damage_detection/dataset/test/images/3_jpeg_jpg.rf.4ab7380d6f0cd508580f448d0f35d1bc.jpg'
    image = cv2.imread(image_path)
    
    # Perform inference
    results = model.predict(image)
    
    # Ensure the results directory exists
    save_dir = 'results/predictions'
    os.makedirs(save_dir, exist_ok=True)
    
    # Iterate through the results and display/save each one
    for result in results:
        result.show()  # Display the result
        result.save(os.path.join(save_dir, 'test_image_predictions.jpg'))  # Save the result

if __name__ == "__main__":
    main()
