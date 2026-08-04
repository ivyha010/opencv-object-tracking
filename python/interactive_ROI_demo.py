import cv2
import time

def create_tracker(tracker_name):
    tracker_name = tracker_name.upper()
    if tracker_name == "CSRT":
        return cv2.legacy.TrackerCSRT_create()
    elif tracker_name == "KCF":
        return cv2.legacy.TrackerKCF_create()
    elif tracker_name == "MOSSE":
        return cv2.legacy.TrackerMOSSE_create()
    elif tracker_name == "MIL":
        return cv2.legacy.TrackerMIL_create()
    else:
        raise ValueError(f"Unknown tracker: {tracker_name}")

def run_tracker(video_path, tracker_name, bbox):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return None

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return None

    # --- Create tracker ---
    tracker = create_tracker(tracker_name)
    tracker.init(frame, bbox)

    frame_count = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        success, box = tracker.update(frame)

        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0

        if success:
            p1 = (int(box[0]), int(box[1]))
            p2 = (int(box[0] + box[2]), int(box[1] + box[3]))
            cv2.rectangle(frame, p1, p2, (0,255,0), 2, 1)

        cv2.putText(frame, f"{tracker_name} | FPS: {fps:.2f} | Frame: {frame_count}",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)

        cv2.imshow(f"Tracking - {tracker_name}", frame)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    cv2.destroyAllWindows()

    # Return average FPS
    elapsed = time.time() - start_time
    avg_fps = frame_count / elapsed if elapsed > 0 else 0
    return avg_fps

def interactive_roi_tracking(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Cannot read video")
        return

    # User draws ROI once
    bbox = cv2.selectROI("Select ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("Select ROI")

    # Cycle through trackers
    trackers = ["CSRT", "KCF", "MOSSE", "MIL"]
    results = {}

    for tname in trackers:
        print(f"\nRunning tracker: {tname}")
        avg_fps = run_tracker(video_path, tname, bbox)
        if avg_fps is not None:
            results[tname] = avg_fps

    # Summary report
    print("\n=== Summary Report ===")
    for tname, fps in results.items():
        print(f"{tname}: Average FPS = {fps:.2f}")

if __name__ == "__main__":
    video_path = "/home/user/Documents/python_data/OTB2015/Basketball.mp4"
    interactive_roi_tracking(video_path)
