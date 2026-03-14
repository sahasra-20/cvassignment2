from vehicle_classifier import VehicleClassifier, CLASS_IDX

classifier = VehicleClassifier("student_model.pth")

image_path = "test.jpg"

pred = classifier.predict(image_path)

print("Predicted index:", pred)
print("Predicted label:", CLASS_IDX[pred])