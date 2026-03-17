from vehicle_classifier import VehicleClassifier, CLASS_IDX

# smallcnn_classifier = VehicleClassifier("smallcnn_model.pth")
image_path = "test2.jpg"
# mobilenet_classifier = VehicleClassifier("mobilenet_model.pth")
classifier = VehicleClassifier(model_path="student_model.pth")  # load your trained weights
idx = classifier.predict("test.jpg")
print(f"Predicted Class Index: {idx}, Label: {CLASS_IDX[idx]}")


# smallcnn_pred = smallcnn_classifier.predict(image_path)
# mobilenet_pred = mobilenet_classifier.predict(image_path)

# print("SmallCNN - Predicted index:", smallcnn_pred)
# print("SmallCNN - Predicted label:", CLASS_IDX[smallcnn_pred])
# print("MobileNet - Predicted index:", mobilenet_pred)
# print("MobileNet - Predicted label:", CLASS_IDX[mobilenet_pred])