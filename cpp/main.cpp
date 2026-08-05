#include <opencv2/opencv.hpp>
#include <opencv2/tracking.hpp>
#include <iostream>
#include <fstream>
#include <filesystem>
#include <chrono>
#include <numeric>
#include <vector>
#include <string>
#include <tuple>  


using namespace cv;
using namespace std;
namespace fs = std::filesystem;

// --- Utility functions ---
Rect2d parseBBox(const string& line) {
    string s = line;
    replace(s.begin(), s.end(), ',', ' ');
    stringstream ss(s);
    double x,y,w,h;
    ss >> x >> y >> w >> h;
    return Rect2d(x,y,w,h);
}

vector<Rect2d> loadGroundtruth(const string& gtFile) {
    vector<Rect2d> bboxes;
    ifstream f(gtFile);
    string line;
    while (getline(f,line)) {
        if(line.empty()) continue;
        bboxes.push_back(parseBBox(line));
    }
    return bboxes;
}

double iou(const Rect2d& a, const Rect2d& b) {
    double interArea = (a & b).area();
    double unionArea = a.area() + b.area() - interArea;
    return unionArea > 0 ? interArea/unionArea : 0.0;
}

double centerError(const Rect2d& a, const Rect2d& b) {
    Point2d ca(a.x + a.width/2.0, a.y + a.height/2.0);
    Point2d cb(b.x + b.width/2.0, b.y + b.height/2.0);
    return norm(ca - cb);
}

// --- Run tracker on one sequence ---
std::tuple<double,double,double> runTracker(const string& trackerType, const string& seqPath, int repeats) {
    string gtFile = seqPath + "/groundtruth_rect.txt";
    vector<Rect2d> gtBoxes = loadGroundtruth(gtFile);

    // Collect image files
    vector<string> imgFiles;
    for (auto& p : fs::directory_iterator(seqPath + "/img")) {
        imgFiles.push_back(p.path().string());
    }
    sort(imgFiles.begin(), imgFiles.end());

    if (imgFiles.empty() || gtBoxes.empty()) {
        cerr << "Skipping " << seqPath << endl;
        return {0.0, 0.0, 0.0};
    }

    double totalMeanIoU = 0.0, totalFPS = 0.0, totalRobustness = 0.0;

    for (int r=0; r<repeats; r++) {
        Ptr<Tracker> tracker;
        if (trackerType == "CSRT") tracker = TrackerCSRT::create();
        else if (trackerType == "KCF") tracker = TrackerKCF::create();
        else if (trackerType == "MIL") tracker = TrackerMIL::create();

        Mat firstFrame = imread(imgFiles[0]);
        if (firstFrame.empty()) {
            cerr << "Empty first frame in " << seqPath << endl;
            continue;
        }

        // Validate initial groundtruth
        Rect2d initBox = gtBoxes[0];
        if (initBox.width <= 0 || initBox.height <= 0 ||
            initBox.x < 0 || initBox.y < 0 ||
            initBox.x + initBox.width > firstFrame.cols ||
            initBox.y + initBox.height > firstFrame.rows) {
            cerr << "Invalid initial groundtruth in " << seqPath << endl;
            continue;
        }

        tracker->init(firstFrame, initBox);

        vector<double> ious;
        int failures = 0;

        auto t0 = chrono::high_resolution_clock::now();
        for (size_t i=0; i<imgFiles.size() && i<gtBoxes.size(); i++) {
            Mat frame = imread(imgFiles[i]);
            if (frame.empty()) {
                cerr << "Empty frame in " << imgFiles[i] << endl;
                continue;
            }

            cv::Rect bbox;   // integer Rect
            bool ok = tracker->update(frame, bbox);

            cv::Rect2d bbox_d(bbox); // convert for IoU

            // Validate groundtruth before IoU
            Rect2d gt = gtBoxes[i];
            if (gt.width <= 0 || gt.height <= 0 ||
                gt.x < 0 || gt.y < 0 ||
                gt.x + gt.width > frame.cols ||
                gt.y + gt.height > frame.rows) {
                cerr << "Invalid groundtruth at frame " << i << " in " << seqPath << endl;
                continue;
            }

            if (ok) {
                double iouVal = iou(bbox_d, gt);
                ious.push_back(iouVal);
                if (iouVal < 0.1) { // failure
                    failures++;
                    tracker->init(frame, gt);
                }
            } else {
                ious.push_back(0.0);
            }
        }
        auto t1 = chrono::high_resolution_clock::now();
        double totalTime = chrono::duration<double>(t1-t0).count();
        double fps = imgFiles.size()/totalTime;
        double meanIoU = accumulate(ious.begin(), ious.end(), 0.0)/ious.size();
        double robustness = (double)failures/imgFiles.size();

        totalMeanIoU += meanIoU;
        totalFPS += fps;
        totalRobustness += robustness;
    }


    double avgFPS = totalFPS / repeats;
    double avgIoU = totalMeanIoU / repeats;
    double avgRobustness = totalRobustness / repeats;

    cout << trackerType << " on " << seqPath << endl;
    cout << "Mean FPS=" << avgFPS
         << " Mean IoU=" << avgIoU
         << " Robustness=" << avgRobustness << endl;

    // return results
    return {avgFPS, avgIoU, avgRobustness};
}

// --- Main with argument parsing ---
int main(int argc, char** argv) {
    // Defaults
    string datasetRoot = "/home/user/Documents/python_data/OTB2015";
    int repeats = 5;
    vector<string> trackers = {"CSRT"};

    // Parse arguments
    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "--dataset_root" && i+1 < argc) {
            datasetRoot = argv[++i];
        } else if (arg == "--repeats" && i+1 < argc) {
            repeats = stoi(argv[++i]);
        } else if (arg == "--tracker") {
            trackers.clear();
            while (i+1 < argc && argv[i+1][0] != '-') {
                trackers.push_back(argv[++i]);
            }
        } else if (arg == "--help") {
            cout << "Usage: ./benchmark "
                 << "[--dataset_root PATH] "
                 << "[--repeats N] "
                 << "[--tracker CSRT KCF MIL]\n";
            return 0;
        }
    }
    
    cout << "Dataset root: " << datasetRoot << endl;
    cout << "Repeats: " << repeats << endl;
    cout << "Trackers: ";
    for (auto& t : trackers) cout << t << " ";
    cout << endl;

    // Global accumulators
    struct Stats { double fps=0, iou=0, robustness=0; int count=0; };
    map<string, Stats> globalStats;

    // Iterate through sequences
    /*
    for (auto& seq : fs::directory_iterator(datasetRoot)) {
        string seqPath = seq.path().string();
        for (auto& t : trackers) {
            // Run tracker and capture results
            // Modify runTracker to return a tuple (fps, iou, robustness)
            auto [fps, iou, rob] = runTracker(t, seqPath, repeats);
            globalStats[t].fps += fps;
            globalStats[t].iou += iou;
            globalStats[t].robustness += rob;
            globalStats[t].count++;
        }
    }
    */


    for (auto& seq : fs::directory_iterator(datasetRoot)) {
        string seqPath = seq.path().string();
        for (auto& t : trackers) {
            auto [fps, iou, rob] = runTracker(t, seqPath, repeats);
            globalStats[t].fps += fps;
            globalStats[t].iou += iou;
            globalStats[t].robustness += rob;
            globalStats[t].count++;
        }
}


    // Print global averages
    cout << "\n=== Global Summary ===\n";
    for (auto& [t, s] : globalStats) {
        cout << t << " across " << s.count << " sequences:\n"
             << "  Avg FPS=" << s.fps / s.count
             << "  Avg IoU=" << s.iou / s.count
             << "  Avg Robustness=" << s.robustness / s.count << endl;
    }

    return 0;
}
