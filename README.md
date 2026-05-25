# Bouldering Pose Estimation Dataset

This repository contains the annotations and framework for a custom indoor bouldering dataset. It provides a robust, high-fidelity 2D dataset for human pose estimation models in complex, gravity-dependent athletic environments. 

Due to the size and format of the media, the raw video files and extracted image frames are hosted externally.

## Dataset Overview

The dataset encapsulates the unique biomechanical challenges of indoor bouldering, focusing heavily on extreme self-occlusion, inverted poses, clothing deformation, and the tracking of distal extremities on artificial holds. 

* **Total Volume:** Approximately 50 minutes of continuous bouldering footage, separated into 100 individual climbs.
* **Video Specifications:** All video sequences are standardized to 20 frames per second (fps). Video durations range from 4.80 seconds to 70.25 seconds, averaging 28.90 seconds.
* **Annotated Frames:** 2,942 distinct frames were extracted using a sparse temporal sampling strategy of 1 fps to maximize pose variance and eliminate redundant static data.
* **Subject Diversity:** The footage features over 20 unique climbers with an approximately equal distribution of male and female subjects, recorded across more than 20 different indoor bouldering facilities. 
* **Real-World Conditions:** Frames containing severe motion blur (common during dynamic lunges or falls) and challenging lighting were intentionally retained to accurately represent real-world bouldering conditions. 

## Repository Structure

The dataset consists of two primary data streams alongside the annotation file. 

* `videos/`: Contains the continuous high-resolution video files (MP4 format) capturing the complete bouldering attempts. These are strictly single-subject videos with no editing or multi-camera transitions.
* `frames/`: Contains the curated subset of 2,942 discrete, uncompressed image frames extracted from the raw videos.
* `annotations.json`: Contains the ground-truth annotations for the extracted frames.

## Annotation Format

The annotations adhere to the industry-standard COCO 17-keypoint skeletal format, encompassing the nose, eyes, ears, shoulders, elbows, wrists, hips, knees, and ankles. Bounding box parameters are included in the JSON and were automatically generated using an object detection framework to provide a standardized detection window. **Please note that because these bounding boxes were generated automatically, they were not manually corrected and may contain errors.**

To systematically handle the severe occlusions common in bouldering, the standard COCO schema was adapted to include a custom four-tier occlusion protocol:

* **`2` (Visible):** The joint is clearly discernible in the camera view and annotated precisely.
* **`3` (High-Confidence Approximation):** The joint is physically obscured, but its exact location is reliably deduced based on rigid anatomical constraints and visible adjacent limbs.
* **`1` (Low-Confidence Approximation):** The joint is heavily obscured, and its spatial location is estimated with a lower degree of confidence based on general biomechanical posture.
* **`0` (Unannotated):** The joint is entirely hidden and cannot be reasonably deduced; these are intentionally omitted to prevent speculative guessing.

## Data Access

The `annotations.json` file is available directly in this repository. To download the accompanying `videos/` and `frames/` directories, please access the externally provided dataset here: 

**https://drive.google.com/file/d/1ByZAg9vDXrD6SGtoKXWOckkMjezaBWA2/view?usp=sharing**
