# **SafeSteps: A Computer Vision-Based Solution for Enhanced Pedestrian Safety**

## Project Overview

SafeSteps is a computer vision system designed to improve pedestrian safety at crosswalks by dynamically controlling traffic signals based on pedestrian movement. It leverages state-of-the-art object detection techniques to enhance pedestrian safety and improve traffic flow.

## Key Features

+ _Accurate Pedestrian Detection:_ Employs YOLO (You Only Look Once) models for real-time identification of pedestrians and crosswalks within a video stream.

+ _Dynamic Traffic Light Control:_ Calculates pedestrian distances from both crosswalk endpoints to determine their intentions (approaching or leaving).

+ _Adaptive Signal Management:_ Adjusts traffic light state (green to yellow to red) based on pedestrian proximity to the crosswalk, prioritizing pedestrian safety while maintaining traffic flow.

+ _Visual Representation_: Generates an output image that clearly depicts:

  1. Detected crosswalk and pedestrians
  
  2. Pink points marking crosswalk edges
  
  3. Blue points representing bounding box centroids
  
  4. White distances indicating proximity to each crosswalk endpoint (purple for end, white for starting edge)

## Getting Started

### Run:

```
cd SafeSteps_ComputerVision\src\safeSteps.py
python safeSteps.py
```

## Expected Output:

![image](https://github.com/saadan1234/SafeSteps_ComputerVision/assets/115701364/a66bffb3-fef5-4cf7-a6ba-4689cdb4d970)


## Further Development

1. Integration with real traffic controllers for practical implementation.

2. Incorporation of additional safety measures, such as audible warnings or pedestrian countdown timers.

3. Training on diverse datasets to enhance pedestrian detection accuracy in various lighting, weather, and clothing conditions.

4. Exploration of alternative object detection algorithms for potential performance improvements.
