# One Step Back, Two Steps Forward: Learning Moves to Recover from SLAM Tracking Failures

## Prerequisites
- [ROS](http://wiki.ros.org/ROS/Installation)
- [MINOS](https://github.com/minosworld/minos)
- [ORBSLAM](https://openslam-org.github.io/orbslam.html)

## Code Guidelines
To set up this project, you need to run MINOS, ORBSLAM, and a ROS node for integration called "merger" simultaneously. To do this, use the bash file and make necessary changes to file paths.

## Code Information
The following files are included in this repository:
- `MINOS_Recovery_SLAM.py`: MINOS agent attempts to regain tracks using the trained network.
- `MINOS_Random_Movement.py`: MINOS agent attempts to regain tracks using random movements.
- `MINOS_Navigation.py`: MINOS agent does not attempt to regain tracks.

Weights Link: https://drive.google.com/file/d/1M9nby4k0wMmCy-EORqSqZ66UVNHwDcbn/view?usp=sharing

[![Demo Video](http://img.youtube.com/vi/Ru5zVv56EQk/0.jpg)](http://www.youtube.com/watch?v=Ru5zVv56EQk)
