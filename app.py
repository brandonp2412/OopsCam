from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import logging
from win10toast import ToastNotifier

logging.getLogger("ultralytics").setLevel(logging.WARNING)

model = YOLO('yolov10n.pt', verbose=False)
# Use 0 for default webcam
cap = cv2.VideoCapture(0) 
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
toaster = ToastNotifier()

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    # Run detection on frame
    results: list[Results] = model(frame, conf=.9)
    annotated_frame = results[0].plot(conf=.9)
    cv2.imshow('YOLO Detection', annotated_frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    for result in results:
        for box in result.boxes:
            name = model.names[int(box.cls)]
            
            if name == 'cell phone':
                toaster.show_toast(
                    "Phone Alert",
                    "Stop using your phone!",
                    duration=5,
                    threaded=True
                )

               




cap.release()
cv2.destroyAllWindows()