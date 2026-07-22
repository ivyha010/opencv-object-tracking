import os
import cv2
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import psutil
import argparse

from trackers.csrt_tracker import CSRTTracker
from trackers.kcf_tracker import KCFTracker
from trackers.mosse_tracker import MOSSETracker
from trackers.mil_tracker import MILTracker
from utils import iou

### Success plot AUC function
def success_auc(iou_scores, thresholds=np.linspace(0,1,101)):
    success_rates = []
    for t in thresholds:
        success_rates.append(np.mean(iou_scores >= t))
    auc = np.mean(success_rates)
    return auc

def load_groundtruth(gt_file):
    bboxes = []
    with open(gt_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            line = line.replace(",", " ")
            parts = line.split()
            if len(parts) >= 4:
                x, y, w, h = map(float, parts[:4])
                bboxes.append([x, y, w, h])
    return np.array(bboxes, dtype=np.float32)

def load_otb_sequence(seq_path, target_id=None):
    img_dir = os.path.join(seq_path, "img")
    frame_files = sorted(os.listdir(img_dir))
    frame_paths = [os.path.join(img_dir, f) for f in frame_files]

    if target_id is None:
        gt_file = os.path.join(seq_path, "groundtruth_rect.txt")
    else:
        gt_file = os.path.join(seq_path, f"groundtruth_rect.{target_id}.txt")

    if not os.path.exists(gt_file):
        print(f"Skipping {seq_path} target {target_id}: groundtruth not found")
        return frame_paths, None

    gt_bboxes = load_groundtruth(gt_file)
    return frame_paths, gt_bboxes


def center_error(pred_box, gt_box):
    px, py, pw, ph = pred_box
    gx, gy, gw, gh = gt_box
    pred_center = (px + pw/2, py + ph/2)
    gt_center = (gx + gw/2, gy + gh/2)
    return np.linalg.norm(np.array(pred_center) - np.array(gt_center))


def run_tracker(tracker_class, seq_path, target_id=None):
    frame_paths, gt_bboxes = load_otb_sequence(seq_path, target_id)
    if gt_bboxes is None or len(gt_bboxes) == 0:
        print(f"Skipping {seq_path} target {target_id}: groundtruth empty or invalid")
        return 0, 0, 0, 0, 0, 0, 0, [], []

    num_frames = min(len(frame_paths), len(gt_bboxes))

    # Initialize tracker with first frame
    first_frame = cv2.imread(frame_paths[0])
    tracker = tracker_class()
    tracker.init(first_frame, tuple(gt_bboxes[0]))

    ious, errors, cpu_logs, mem_logs = [], [], [], []
    failures = 0
    process = psutil.Process(os.getpid())

    t0 = time.time()
    for i in range(num_frames):
        frame = cv2.imread(frame_paths[i])
        success, box = tracker.update(frame)

        if success:
            pred_box = [int(v) for v in box]
            gt_box = gt_bboxes[i]
            iou_val = iou(pred_box, gt_box)
            ious.append(iou_val)
            errors.append(center_error(pred_box, gt_box))

            if iou_val < 0.1:
                failures += 1
                tracker.init(frame, tuple(gt_box))
        else:
            ious.append(0.0)
            errors.append(9999)

        cpu_logs.append(process.cpu_percent(interval=None))
        mem_logs.append(process.memory_info().rss / (1024*1024))
    t1 = time.time()

    total_time = t1 - t0
    mean_fps = num_frames / total_time if total_time > 0 else 0
    median_fps = mean_fps

    mean_iou = np.mean(ious) if ious else 0
    robustness = failures / num_frames if num_frames > 0 else 0
    avg_cpu = np.mean(cpu_logs) if cpu_logs else 0
    peak_mem = np.max(mem_logs) if mem_logs else 0
    auc = success_auc(np.array(ious)) if ious else 0

    return mean_fps, median_fps, mean_iou, robustness, avg_cpu, peak_mem, auc, ious, errors


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark trackers on OTB sequences")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Number of times to repeat each tracker per sequence (default=1)")
    args = parser.parse_args()

    dataset_root = "/home/user/Documents/python_data/OTB2015"
    sequences = os.listdir(dataset_root)

    trackers = {
        "CSRT": CSRTTracker,
        "KCF": KCFTracker,
        "MOSSE": MOSSETracker,
        "MIL": MILTracker
    }

    results = []
    all_success_curves = {}   ### store per-frame IoUs for success plot
    all_center_errors = {}    ### store per-frame center errors for precision plot

    for seq in sequences:
        seq_path = os.path.join(dataset_root, seq)
        targets = [1, 2] if seq in ["Jogging", "Skating2", "Human4"] else [None]

        for target_id in targets:
            frames, gt_bboxes = load_otb_sequence(seq_path, target_id)
            if gt_bboxes is None:
                continue

            seq_name = seq if target_id is None else f"{seq}-{target_id}"

            for name, tracker_class in trackers.items():
                mean_fps_list, median_fps_list, iou_list, rob_list, cpu_list, mem_list, auc_list, iou_frames_all, error_frames_all = [], [], [], [], [], [], [], [], []
                for _ in range(args.repeats):
                    mean_fps, median_fps, mean_iou, robustness, avg_cpu, peak_mem, auc, iou_frames, errors = run_tracker(tracker_class, seq_path, target_id)
                    mean_fps_list.append(mean_fps)
                    median_fps_list.append(median_fps)
                    iou_list.append(mean_iou)
                    rob_list.append(robustness)
                    cpu_list.append(avg_cpu)
                    mem_list.append(peak_mem)
                    auc_list.append(auc)
                    iou_frames_all.extend(iou_frames)
                    error_frames_all.extend(errors)

                mean_fps = np.mean(mean_fps_list)
                median_fps = np.mean(median_fps_list)
                mean_iou = np.mean(iou_list)
                robustness = np.mean(rob_list)
                avg_cpu = np.mean(cpu_list)
                peak_mem = np.mean(mem_list)
                auc = np.mean(auc_list)

                results.append({
                    "Sequence": seq_name,
                    "Tracker": name,
                    "Mean FPS": mean_fps,
                    "Median FPS": median_fps,
                    "Mean IoU": mean_iou,
                    "Robustness": robustness,
                    "CPU %": avg_cpu,
                    "Memory MB": peak_mem,
                    "Success AUC": auc
                })
                print(f"{seq_name} - {name}: Mean FPS={mean_fps:.2f}, Median FPS={median_fps:.2f}, IoU={mean_iou:.3f}, "
                      f"Robustness={robustness:.3f}, CPU={avg_cpu:.1f}%, Mem={peak_mem:.1f}MB, AUC={auc:.3f} "
                      f"(averaged over {args.repeats} runs)")

                if name not in all_success_curves:
                    all_success_curves[name] = []
                if name not in all_center_errors:
                    all_center_errors[name] = []
                all_success_curves[name].extend(iou_frames_all)
                all_center_errors[name].extend(error_frames_all)

    df = pd.DataFrame(results)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)
    df.to_csv("results/benchmark.csv", index=False)
    print("\nBenchmark complete. Results saved to results/benchmark.csv")

## --- Plotting ---
avg_results = df.groupby("Tracker")[["Mean FPS", "Median FPS", "Mean IoU", "Robustness", "CPU %", "Memory MB", "Success AUC"]].mean().reset_index()
avg_results = avg_results.round(2)
print("\n=== Average Metrics Across All Sequences ===")
print(avg_results.to_string(index=False))

### Mean FPS bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Mean FPS"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average Mean FPS per Tracker")
plt.ylabel("Frames Per Second")
plt.savefig("figures/mean_fps_comparison.png")
plt.close()

### Median FPS bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Median FPS"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average Median FPS per Tracker")
plt.ylabel("Frames Per Second")
plt.savefig("figures/median_fps_comparison.png")
plt.close()

# IoU bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Mean IoU"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average IoU per Tracker")
plt.ylabel("Mean IoU")
plt.savefig("figures/iou_comparison.png")
plt.close()

# Robustness bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Robustness"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average Robustness (failures/frame)")
plt.ylabel("Failure Rate")
plt.savefig("figures/robustness_comparison.png")
plt.close()

# CPU usage bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["CPU %"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average CPU Usage per Tracker")
plt.ylabel("CPU %")
plt.savefig("figures/cpu_comparison.png")
plt.close()

# Memory usage bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Memory MB"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average Memory Usage per Tracker")
plt.ylabel("Memory (MB)")
plt.savefig("figures/memory_comparison.png")
plt.close()

# Success AUC bar chart
plt.figure(figsize=(8, 6))
plt.bar(avg_results["Tracker"], avg_results["Success AUC"],
        color=["#4caf50", "#2196f3", "#ff9800", "#9c27b0"])
plt.title("Average Success AUC per Tracker")
plt.ylabel("AUC (Success Plot)")
plt.savefig("figures/success_auc_comparison.png")
plt.close()

# Success Plot (line graph)
thresholds = np.linspace(0, 1, 201)
plt.figure(figsize=(8, 6))
for tracker, ious in all_success_curves.items():
    success_rates = [np.mean(np.array(ious) >= t) for t in thresholds]
    plt.plot(thresholds, success_rates, label=tracker)

plt.title("Success Plot (IoU Threshold vs Success Rate)")
plt.xlabel("IoU Threshold")
plt.ylabel("Success Rate")
plt.legend()
plt.grid(True)
plt.savefig("figures/success_plot.png")
plt.close()

### Precision Plot (line graph)
thresholds = np.arange(0, 51, 1)  # pixel error thresholds
plt.figure(figsize=(8, 6))
for tracker, errors in all_center_errors.items():
    precision = [np.mean(np.array(errors) <= t) for t in thresholds]
    plt.plot(thresholds, precision, label=tracker)

plt.title("Precision Plot (Center Error Threshold vs Precision)")
plt.xlabel("Center Error Threshold (pixels)")
plt.ylabel("Precision")
plt.legend()
plt.grid(True)
plt.savefig("figures/precision_plot.png")
plt.close()

print("Figures saved to figures/mean_fps_comparison.png, figures/median_fps_comparison.png, "
      "figures/iou_comparison.png, figures/robustness_comparison.png, figures/cpu_comparison.png, "
      "figures/memory_comparison.png, figures/success_auc_comparison.png, figures/success_plot.png, "
      "and figures/precision_plot.png")
