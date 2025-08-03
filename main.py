from ultralytics import YOLO

model = YOLO("yolov8x")  # Load a pre-trained YOLOv8 model

results = model.predict("input_videos/video_1.mp4", save=True)  # Use webcam as source and display results
print(results)  # Print the results
print("------------------------")
for box in results[0].boxes:
    print(f"Box: {box}")

