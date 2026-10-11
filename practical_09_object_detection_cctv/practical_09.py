from pathlib import Path
import cv2

try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "Ultralytics is not installed. Run: pip install ultralytics"
    )

BASE = Path(__file__).resolve().parent
input_path = BASE / "data" / "input.mp4"
output_path = BASE / "outputs" / "detected_output.mp4"

if not input_path.exists():
    print("CCTV input video not found.")
    print(f"Place a video at: {input_path}")
    print("Then run this program again.")
    raise SystemExit(0)

model = YOLO("yolo11n.pt")
cap = cv2.VideoCapture(str(input_path))

if not cap.isOpened():
    raise SystemExit("Could not open the input video.")

fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

frame_count = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break

    result = model(frame, verbose=False)[0]
    annotated = result.plot()
    writer.write(annotated)
    frame_count += 1

cap.release()
writer.release()

print(f"Processed frames: {frame_count}")
print(f"Saved annotated video: {output_path}")
