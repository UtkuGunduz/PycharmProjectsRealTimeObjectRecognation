import cv2 as cv

directory_path = "/Users/utkugunduz/Documents/Akdeniz Üniversitesi/Bilgisayar Programcılığı/2. Sınıf/Sistem Analiz ve Tasarımı I/Fİnal Nesne Tanıma Proje/Models"
model_file = directory_path + "/MobileNetSSD_deploy.caffemodel"
config_file = directory_path + "/MobileNetSSD_deploy.prototxt"

object_labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

neural_net = cv.dnn.readNetFromCaffe(config_file, model_file)

camera_capture = cv.VideoCapture(0)
while True:
    ret, frame = camera_capture.read()
    if not ret:
        break
    height, width = frame.shape[:2]
    blob_image = cv.dnn.blobFromImage(frame, 0.007843, (350, 350), (127.5, 127.5, 127.5), True, False)
    neural_net.setInput(blob_image)
    neural_output = neural_net.forward()
    for detection in neural_output[0, 0, :, :]:
        confidence = float(detection[2])
        label_index = int(detection[1])
        if confidence > 0.4:
            left_coord = detection[3] * width
            top_coord = detection[4] * height
            right_coord = detection[5] * width
            bottom_coord = detection[6] * height

            cv.rectangle(frame, (int(left_coord), int(top_coord)), (int(right_coord), int(bottom_coord)), (255, 0, 0), thickness=3)
            cv.putText(frame, "confidence: %.2f, %s" % (confidence, object_labels[label_index]), (int(left_coord) - 10, int(top_coord) - 5), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, 8)
    cv.imshow("object-recognation", frame)
    key = cv.waitKey(10)
    if key == 27:
        break
cv.waitKey(0)
