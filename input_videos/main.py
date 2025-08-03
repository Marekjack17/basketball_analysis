from ultralytics import YOLO

model = YOLO("yolov8x")  # Load a pre-trained YOLOv8 model

results = model.predict(source="input_videos/video_1.mp4", save=True)  # Predict on a video file

print(results)  # Print the results of the prediction
print("------------------------")
for box in results[0].boxes:
    print(box)