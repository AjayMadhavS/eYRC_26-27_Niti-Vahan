'''
*****************************************************************************************
*
*  ===============================================
*     Niti Vahan (NV) Theme of eYRC 2026-27
*  ===============================================
*
*  This script is intended for implementation of Task 1C of Niti Vahan (NV) Theme.
*
*  Filename:         lane_detection.py
*  Created:          2026
*  Last Modified:
*  Author:           e-Yantra Team
*
*  You are ONLY allowed to write your code inside the block marked
*  "ADD YOUR IMPLEMENTATION HERE". Do not change anything outside it - the
*  evaluation script relies on the rest of this file staying as it is.
*
*****************************************************************************************
'''

# Team ID:          < Team-ID >
# Author List:      < Names of the team members who worked on this file, comma separated >
# Filename:         lane_detection.py
# Functions:        detect_lane
# Global variables: g_last_lane


####################### IMPORT MODULES #######################
import argparse
import json
import os

import cv2
import numpy as np
##############################################################

# The only three values "lane" is allowed to take.
LANE_LEFT = "left"
LANE_RIGHT = "right"
LANE_UNKNOWN = "unknown"
VALID_LANES = (LANE_LEFT, LANE_RIGHT, LANE_UNKNOWN)


##############################################################
############### ADD YOUR IMPLEMENTATION HERE #################
##############################################################

def detect_lane(frame):
    '''
    Purpose:
    ---
    Detect the lane in a single frame and report where the centre of the lane
    is, and which of the two lanes the vehicle is currently in.
    '''
    global g_last_lane
    global g_last_lane_age

    Y_EVAL = 360
    LANE_WIDTH = 315
    frame_height, frame_width = frame.shape[:2]
    if frame_width < 2 or frame_height < 2:
        return {"center_x": -1, "lane": LANE_UNKNOWN}

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    yellow_mask = cv2.inRange(
        hsv, np.array([16, 85, 65]), np.array([40, 255, 255]))
    white_mask = cv2.inRange(
        hsv, np.array([0, 0, 170]), np.array([180, 75, 255]))

    yellow = _estimate_lane_mark(yellow_mask, Y_EVAL, "yellow")
    white = _estimate_lane_mark(white_mask, Y_EVAL, "white")
    yellow_x = (yellow["x"] if yellow is not None
                and yellow["confidence"] >= 0.10 else None)
    white_x = (white["x"] if white is not None
               and white["confidence"] >= 0.10 else None)

    
    lane = LANE_UNKNOWN

    if white_x is not None:
        if yellow_x is not None:
            pair_width = abs(white_x - yellow_x)

            if 180 <= pair_width <= 560:
                lane = (
                    LANE_LEFT
                    if yellow_x < white_x
                    else LANE_RIGHT
                )
            else:
                lane = (
                    LANE_LEFT
                    if white_x >= frame_width / 2.0
                    else LANE_RIGHT
                )
        else:
            lane = (
                LANE_LEFT
                if white_x >= frame_width / 2.0
                else LANE_RIGHT
            )

    elif yellow_x is not None:
        lane = (
            LANE_LEFT
            if yellow_x < frame_width / 2.0
            else LANE_RIGHT
        )

    if lane in (LANE_LEFT, LANE_RIGHT):
        g_last_lane = lane
        g_last_lane_age = 0
    else:
        g_last_lane_age += 1
        if (
            g_last_lane_age <= 5
            and g_last_lane in (LANE_LEFT, LANE_RIGHT)
        ):
            lane = g_last_lane


    center_x = -1
    if (lane == LANE_LEFT and yellow_x is not None and white_x is not None
            and yellow_x < white_x):
        width = white_x - yellow_x
        if 180 <= width <= 560:
            center_x = int(round((yellow_x + white_x) / 2.0))
    elif (lane == LANE_RIGHT and yellow_x is not None and white_x is not None
          and white_x < yellow_x):
        width = yellow_x - white_x
        if 180 <= width <= 560:
            center_x = int(round((yellow_x + white_x) / 2.0))

    if center_x < 0 and white_x is not None and lane in (LANE_LEFT, LANE_RIGHT):
        offset = LANE_WIDTH / 2.0
        estimate = white_x - offset if lane == LANE_LEFT else white_x + offset
        if 0 <= estimate < frame_width:
            center_x = int(round(estimate))

    if center_x < 0 and yellow_x is not None and lane in (LANE_LEFT, LANE_RIGHT):
        yellow_is_near_side = (
            (lane == LANE_LEFT and yellow_x < frame_width / 2.0)
            or (lane == LANE_RIGHT and yellow_x > frame_width / 2.0))
        if yellow_is_near_side:
            offset = LANE_WIDTH / 2.0
            estimate = yellow_x + offset if lane == LANE_LEFT else yellow_x - offset
            if 0 <= estimate < frame_width:
                center_x = int(round(estimate))

    if not 0 <= center_x < frame_width:
        center_x = -1
    return {"center_x": center_x, "lane": lane}


# ------------------------------------------------------------------
# Add any helper functions and global variables you need below this
# comment, and keep them ABOVE the "END OF YOUR IMPLEMENTATION" line.
# They must be called from detect_lane() - the evaluation script only
# ever calls that one function. List them in the file header too.
# ------------------------------------------------------------------

# Lane state is retained only briefly while the dashed divider is not visible.
g_last_lane = LANE_UNKNOWN
g_last_lane_age = 100


def _estimate_lane_mark(mask, target_y, colour):
    height, width = mask.shape[:2]
    roi_top = max(0, target_y - 105)
    roi_bottom = min(height, target_y + 86)
    if roi_bottom - roi_top < 20:
        return None

    roi = mask[roi_top:roi_bottom, :]
    opened = cv2.morphologyEx(
        roi, cv2.MORPH_OPEN, np.ones((3, 3), dtype=np.uint8))
    filtered = cv2.morphologyEx(
        opened, cv2.MORPH_CLOSE, np.ones((9, 3), dtype=np.uint8))
    contours, _ = cv2.findContours(
        filtered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    sample_y = []
    sample_x = []
    sample_support = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        contour_area = cv2.contourArea(contour)
        direction_x, direction_y, _, _ = cv2.fitLine(
            contour, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        contour_points = contour.reshape(-1, 2).astype(np.float32)
        normal_x, normal_y = -direction_y, direction_x
        cross_track = contour_points[:, 0] * normal_x + contour_points[:, 1] * normal_y
        contour_thickness = float(np.percentile(
            np.abs(cross_track - np.median(cross_track)), 90) * 2.0)

        # Lane paint is vertically elongated in this lower-road ROI. This
        # rejects crosswalk bars and stop lines before fitting any positions.
        if (box_height < 7 or abs(direction_y) < 0.12
                or (colour == "white" and box_width > 90)
                or contour_thickness > 105 or contour_area < 3):
            continue

        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y
        inside = np.zeros((box_height, box_width), dtype=np.uint8)
        cv2.drawContours(inside, [local_contour], -1, 255, thickness=-1)
        component = cv2.bitwise_and(
            filtered[y:y + box_height, x:x + box_width], inside)
        ys, xs = np.nonzero(component)
        if len(xs) < (16 if colour == "white" else 14):
            continue

        max_band_width = 80 if colour == "yellow" else 38
        for band in np.unique(ys // 6):
            in_band = (ys // 6) == band
            band_x = xs[in_band]
            if (len(band_x) < 2
                    or np.percentile(band_x, 90)
                    - np.percentile(band_x, 10) > max_band_width):
                continue
            sample_y.append(roi_top + y + float(np.median(ys[in_band])))
            sample_x.append(x + float(np.median(band_x)))
            sample_support.append(len(band_x))

    if len(sample_y) < 3:
        return None

    ys = np.asarray(sample_y, dtype=np.float64)
    xs = np.asarray(sample_x, dtype=np.float64)
    support = np.asarray(sample_support, dtype=np.float64)
    near_target = np.abs(ys - target_y) <= 82
    ys = ys[near_target]
    xs = xs[near_target]
    support = support[near_target]
    accepted_pixels = int(np.sum(support))
    if len(ys) < 3:
        return None

    y_delta = ys[:, None] - ys[None, :]
    x_delta = xs[:, None] - xs[None, :]
    pair_mask = (np.abs(y_delta) >= 8) & (np.abs(y_delta) <= 150)
    slopes = x_delta[pair_mask] / y_delta[pair_mask]
    if slopes.size == 0:
        return None

    slope = float(np.median(slopes))
    intercept = float(np.median(xs - slope * (ys - target_y)))
    residual = xs - (slope * (ys - target_y) + intercept)
    residual_center = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_center)))
    tolerance = max(9.0, min(22.0, 2.8 * 1.4826 * mad))
    inliers = np.abs(residual - residual_center) <= tolerance
    if np.count_nonzero(inliers) < 3:
        return None

    fit_y = ys[inliers]
    fit_x = xs[inliers]
    fit_support = support[inliers]
    degree = 2 if len(fit_y) >= 6 and np.ptp(fit_y) >= 42 else 1
    fit_weights = np.sqrt(fit_support) / (
        1.0 + np.abs(fit_y - target_y) / 90.0)
    coefficients = np.polyfit(
        (fit_y - target_y) / 100.0, fit_x, degree, w=fit_weights)
    for _ in range(2):
        fit_residual = fit_x - np.polyval(
            coefficients, (fit_y - target_y) / 100.0)
        fit_median = float(np.median(fit_residual))
        fit_mad = float(np.median(np.abs(fit_residual - fit_median)))
        keep = np.abs(fit_residual - fit_median) <= max(7.0, 2.8 * fit_mad)
        if np.count_nonzero(keep) < max(3, degree + 1):
            break
        fit_y = fit_y[keep]
        fit_x = fit_x[keep]
        fit_support = fit_support[keep]
        degree = 2 if len(fit_y) >= 6 and np.ptp(fit_y) >= 42 else 1
        fit_weights = np.sqrt(fit_support) / (
            1.0 + np.abs(fit_y - target_y) / 90.0)
        coefficients = np.polyfit(
            (fit_y - target_y) / 100.0, fit_x, degree,
            w=fit_weights)

    estimate = float(np.polyval(coefficients, 0.0))
    support_span = float(np.ptp(fit_y))
    nearest_support = float(np.min(np.abs(fit_y - target_y)))
    required_pixels = 26 if colour == "white" else 20
    required_span = 18 if colour == "white" else 16
    plausible_x = (0 <= estimate < width if colour == "white"
                   else -80 <= estimate <= width + 80)
    if (accepted_pixels < required_pixels or len(fit_y) < 3
            or support_span < required_span or nearest_support > 58
            or not plausible_x):
        return None

    confidence = min(1.0, accepted_pixels / 70.0)
    confidence *= min(1.0, support_span / 50.0)
    return {"x": estimate, "confidence": confidence,
            "support": accepted_pixels, "span": support_span}

##############################################################
################ END OF YOUR IMPLEMENTATION ##################
##############################################################


#################### DO NOT EDIT BELOW THIS LINE ####################

def validate_result(result, frame_index):
    '''
    Purpose:
    ---
    Check that detect_lane() returned the expected structure and normalise it,
    so that a malformed return is reported here instead of silently scoring
    zero during evaluation.

    Input Arguments:
    ---
    result :          [ object ]      whatever detect_lane() returned
    frame_index :     [ int ]         index of the frame, used in error messages

    Returns:
    ---
    clean :           [ dict ]        {"center_x": int, "lane": str}
    '''
    where = "detect_lane() on frame {}".format(frame_index)

    if not isinstance(result, dict):
        raise TypeError("{} must return a dict, got {}".format(where, type(result).__name__))

    missing = {"center_x", "lane"} - set(result.keys())
    if missing:
        raise ValueError("{} is missing the key(s): {}".format(where, ", ".join(sorted(missing))))

    center_x = result["center_x"]
    if isinstance(center_x, bool) or not isinstance(center_x, (int, float, np.integer, np.floating)):
        raise TypeError("{} returned center_x of type {}, expected a number".format(
            where, type(center_x).__name__))
    center_x = int(round(float(center_x)))

    lane = result["lane"]
    if not isinstance(lane, str):
        raise TypeError("{} returned lane of type {}, expected a string".format(
            where, type(lane).__name__))
    lane = lane.strip().lower()
    if lane not in VALID_LANES:
        raise ValueError("{} returned lane = '{}', expected one of {}".format(
            where, result["lane"], ", ".join(VALID_LANES)))

    return {"center_x": center_x, "lane": lane}


def draw_overlay(frame, result):
    '''
    Purpose:
    ---
    Draw the detected lane centre and lane label on a copy of the frame.
    This is where display code belongs - never inside detect_lane().

    Input Arguments:
    ---
    frame :   [ numpy.ndarray ]   the frame that was passed to detect_lane()
    result :  [ dict ]            the validated result for that frame

    Returns:
    ---
    canvas :  [ numpy.ndarray ]   a copy of the frame with the overlay drawn
    '''
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # frame centre, for reference - roughly where the vehicle is pointing
    cv2.line(canvas, (width // 2, height), (width // 2, height - 40), (128, 128, 128), 1)

    center_x = result["center_x"]
    if 0 <= center_x < width:
        cv2.line(canvas, (center_x, height), (center_x, height // 2), (0, 0, 255), 2)
        cv2.circle(canvas, (center_x, height - 10), 5, (0, 0, 255), -1)

    label = "lane: {}   center_x: {}".format(result["lane"], center_x)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return canvas


def process_video(video_path, show=False):
    '''
    Purpose:
    ---
    Read a video frame by frame, hand each frame to detect_lane() and collect
    the results.

    Input Arguments:
    ---
    video_path :  [ str ]     path to the video file
    show :        [ bool ]    if True, display the overlay while processing

    Returns:
    ---
    results :     [ list ]    one dict per frame:
                                {"frame": int, "center_x": int, "lane": str}
    '''
    if not os.path.isfile(video_path):
        raise FileNotFoundError("no such video file: {}".format(video_path))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise IOError("OpenCV could not open the video: {}".format(video_path))

    window = "Task 1C - {}".format(os.path.basename(video_path))
    results = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            # A copy is passed in, so anything drawn inside detect_lane() cannot
            # corrupt the frame used for display.
            result = validate_result(detect_lane(frame.copy()), frame_index)
            results.append({"frame": frame_index, **result})

            if show:
                cv2.imshow(window, draw_overlay(frame, result))
                if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
                    break

            frame_index += 1
    finally:
        capture.release()
        if show:
            cv2.destroyAllWindows()

    if not results:
        raise IOError("no frames could be read from: {}".format(video_path))

    return results


def summarise(video_path, results):
    '''
    Purpose:
    ---
    Print a one-line-per-video summary, so you can see at a glance whether the
    detector is returning anything sensible.

    Input Arguments:
    ---
    video_path :  [ str ]     path to the video that was processed
    results :     [ list ]    output of process_video()

    Returns:
    ---
    None
    '''
    total = len(results)
    counts = {lane: 0 for lane in VALID_LANES}
    for entry in results:
        counts[entry["lane"]] += 1
    not_found = sum(1 for entry in results if entry["center_x"] < 0)

    print("{:<28} {:>5} frames | left {:>5} | right {:>5} | unknown {:>5} | no centre {:>5}".format(
        os.path.basename(video_path), total,
        counts[LANE_LEFT], counts[LANE_RIGHT], counts[LANE_UNKNOWN], not_found))


def expand_videos(paths):
    '''
    Purpose:
    ---
    Turn the command-line arguments into a list of video files, accepting a
    FOLDER as well as individual files.

    A folder is the portable way to say "all the clips": Windows shells do not
    expand public/*.mp4 the way bash does - cmd and PowerShell hand the
    pattern through verbatim and the script would look for a file literally
    named "*.mp4". python lane_detection.py public behaves the same on every
    platform.

    Input Arguments:
    ---
    paths :   [ list ]    the raw command-line arguments

    Returns:
    ---
    videos :  [ list ]    paths to individual video files, folders expanded
    '''
    videos = []
    for raw in paths:
        if os.path.isdir(raw):
            found = sorted(f for f in os.listdir(raw) if f.lower().endswith(".mp4"))
            if not found:
                raise FileNotFoundError("no .mp4 files in the folder: {}".format(raw))
            videos.extend(os.path.join(raw, f) for f in found)
        else:
            videos.append(raw)
    return videos


def main():
    parser = argparse.ArgumentParser(
        description="Task 1C - run your lane detector over one or more videos.")
    parser.add_argument("videos", nargs="+",
                        help="video file(s), or a folder holding them "
                             "(e.g. 'public')")
    parser.add_argument("--show", action="store_true",
                        help="display the detection overlay while processing (press q to stop)")
    parser.add_argument("--out", metavar="FILE",
                        help="write the per-frame results to this JSON file")
    args = parser.parse_args()

    all_results = {}
    for video_path in expand_videos(args.videos):
        results = process_video(video_path, show=args.show)
        summarise(video_path, results)
        all_results[os.path.basename(video_path)] = results

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(all_results, handle, indent=2)
        print("\nresults written to {}".format(args.out))


if __name__ == "__main__":
    main()