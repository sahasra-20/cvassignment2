from vehicle_classifier import VehicleClassifier, CLASS_IDX

smallcnn_classifier = VehicleClassifier("smallcnn_model.pth")
mobilenet_classifier = VehicleClassifier("mobilenet_model.pth")

image_path = "test.jpg"

smallcnn_pred = smallcnn_classifier.predict(image_path)
mobilenet_pred = mobilenet_classifier.predict(image_path)

print("SmallCNN - Predicted index:", smallcnn_pred)
print("SmallCNN - Predicted label:", CLASS_IDX[smallcnn_pred])
print("MobileNet - Predicted index:", mobilenet_pred)
print("MobileNet - Predicted label:", CLASS_IDX[mobilenet_pred])