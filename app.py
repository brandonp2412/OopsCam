from ultralytics import YOLO
from ultralytics.engine.results import Results
import cv2
import logging
from desktop_notifier import DesktopNotifierSync

logging.getLogger("ultralytics").setLevel(logging.WARNING)

model = YOLO("yolov10n.pt", verbose=False)
cap = cv2.VideoCapture(0)  # Use 0 for default webcam
notifier = DesktopNotifierSync()
cell_phone_id = 67

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection on frame
    results: list[Results] = model(frame, conf=0.5, classes=[cell_phone_id])
    annotated_frame = results[0].plot()
    cv2.imshow("YOLO Detection", annotated_frame)

    # Break loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    for result in results:
        if len(result.boxes) > 0:
            notifier.send(
                title="Stop! Get off that phone!",
                message="I see you scrolling you sneaky bastard",
            )


cap.release()
cv2.destroyAllWindows()
