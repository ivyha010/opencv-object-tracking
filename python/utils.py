import cv2
import time

def calculate_fps(start_time, frame_count):
    elapsed_time = time.time() - start_time
    return frame_count / elapsed_time if elapsed_time > 0 else 0

def draw_bbox(frame, bbox, color=(0,255,0)):
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(frame, (x,y), (x+w,y+h), color, 2)

def save_video(output_path, frames, fps, size):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)
    for f in frames:
        out.write(f)
    out.release()

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0]+boxA[2], boxB[0]+boxB[2])
    yB = min(boxA[1]+boxA[3], boxB[1]+boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]
    unionArea = boxAArea + boxBArea - interArea

    return interArea / unionArea if unionArea > 0 else 0

