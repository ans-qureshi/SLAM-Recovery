# One Step Back, Two Steps Forward: Learning Moves to Recover from SLAM Tracking Failures
Tracking failure in state-of-the-art visual SLAM has been reported to be frequent and hampers real-world deployment of SLAM solutions. Very recently, efforts have been made to avoid tracking failure using various methods (e.g. using deep reinforcement learning to plan a path where the risk of tracking failure is minimal). The results of those approaches are encouraging but are far from producing a failure-free visual system. Failure is inevitable in vision-based systems. Therefore, developing recovery (post-failure) solutions might be a better approach in developing more reliable systems. To this end, we propose a novel, easily trainable, deep recovery maneuver algorithm. Instead of predicting tracking failures, our algorithm predicts the back stepping move, post failure, which helps regain tracking.

To view our paper, please visit the following:
Qureshi, A. et al. (2024). *One Step Back, Two Steps Forward: Learning Moves to Recover from SLAM Tracking Failures*. [DOI:10.1080/01691864.2024.2319144](https://doi.org/10.1080/01691864.2024.2319144)
# BibTex
```
@article{hussain2024one,
  title={One step back, two steps forward: learning moves to recover from SLAM tracking failures},
  author={Hussain Qureshi, Ans and Latif Anjum, Muhammmad and Hussain, Wajahat and Muddassar, Usama and Abbasi, Sohail},
  journal={Advanced Robotics},
  pages={1--16},
  year={2024},
  publisher={Taylor \& Francis}
}
```

## Video

https://github.com/ans-qureshi/SLAM-Recovery/assets/38855178/e64633c1-7b4c-4190-90cb-2acc468d2d70


# Our MINOS+SLAM Interface

Our framework facilitates the seamless integration of visual SLAM with a simulator (`MINOS` in this instance). Within this setup, the agent navigates within the simulator while visual SLAM (specifically ORB-SLAM) runs concurrently on the agent's front-view camera. This setup enables the training and testing of SLAM in an active setting. 

## MINOS+SLAM Integration (Step-by-Step)

### Pre-requisites

```bash
sudo apt-get install python3.5-dev && sudo apt-get install python3-tk && sudo apt-get install build-essential libxi-dev libglu1-mesa-dev libglew-dev libopencv-dev libvips && sudo apt install git && sudo apt install curl && libboost-all-dev
```
- [ROS Kinetic](http://wiki.ros.org/kinetic/Installation)
- [MINOS](https://github.com/minosworld/minos)
- [ORBSLAM](https://openslam-org.github.io/orbslam.html)

## Install ORB-SLAM (as ROS node):

- At home type: `git clone “https://github.com/raulmur/ORB_SLAM”`
- After download is complete, build third party packages, build `g2o`.

```bash
sh
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
- Build `DBoW2`. Go into ```Thirdparty/DBoW2/``` and run:
```bash
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
A few changes need to be done before building `ORB_SLAM`. After compiling thirdparty `g2o` and `DBoW2`, before building `ORB_SLAM`:
- In `src/ORBextractor.cc` include OpenCV library: ```#include <opencv2/opencv.hpp>```
- Remove `opencv2` dependency from ```manifest.xml```
- In ```CmakeList.txt```, add `lboost` as target link library:
Replace:
```
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so
)
```
with 
```
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${EIGEN3_LIBS}
    ${PROJECT_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so
    ${PROJECT_SOURCE_DIR}/Thirdparty/g2o/lib/libg2o.so-lboost_system
)
```
- Install `eigen` from here. Download the debian file and install using:
```bash
sudo dpkg -i libeigen3-dev_3.2.0-8_all.deb
```
- Before building `ORB-SLAM` run this in terminal:
```bash
export ROS_PACKAGE_PATH=${ROS_PACKAGE_PATH}:/home/$(hostname)
```
Then run:
```bash
roslaunch ORB_SLAM ExampleGroovyOrNewer.launch
```
In the file `ORB_SLAM/SRC/TRACKING.CC`, on line 163 use following line:
```
ros::Subscriber sub = nodeHandler.subscribe("/usb_cam/image_raw", 1, &Tracking::GrabImage, this);
```

##Install `MINOS`
```bash
curl -o- https://raw.githubusercontent.com/creationix/nvm/v0.33.7/install.sh | bash
source ~/.bashrc
nvm install 8.11.3
nvm alias default 8.11.3
```
OR
```
nvm install v10.13.0
nvm alias default 10.13.0
```
Build the `MINOS` server modules inside the server directory by:
```bash
npm install -g yarn
yarn install
```
OR (not recommended):
```bash
npm install
```
This process will download and compile all server module dependencies and might take a few minutes.

Install the `MINOS` Python module by running `pip3 install -e` in the root of the repository or `pip3 install -e . -r requirements.txt`.

Before running `MINOS`, copy the Materport3D and SUNCG datasets in the work folder. Check that everything works by running the interactive client through:
```bash
python3 -m minos.tools.pygame_client
python3 -m minos.tools.pygame_client --dataset mp3d --scene_ids 17DRP5sb8fy --env_config pointgoal_mp3d_s –save_png --depth
```
Invoked from the root of the MINOS repository. You should see a live view which you can control with the W/A/S/D keys and the arrow keys. This client can be configured through various command line arguments. Run with the `--help` argument for an overview and try some of these other examples.
##Interfacing MINOS and ORB-SLAM Together

For merging purpose, we need to save simulator frames in a folder of our choice. Go to `minos/minos/lib/Simulator.py` and make the following changes:

Replace lines 98-102 with (with your own folder address):
```python
if 'logdir' in params:
    self._logdir = '/home/romi/frames'
else:
    self._logdir = '/home/romi/frames'
```

Replace lines 422-428 with (with your own folder address):
```python
if self.params.get(‘save_png’):
    if image is None:
        image = Image.frombytes(mode,(data.shape[0], data.shape[1]),data)
    time.sleep(0.06)
    image.save(‘/home/romi/frames/color_.jpg’)
```
Create `ROS` Node (`merger.cpp` provided above).
This ROS node is important for communication between MINOS and ORB-SLAM. Simply paste the attached ROS node (merger.cpp) in catkin_ws/src and do necessary changes to CmakeList.txt by uncommenting the following:
```C++
add_compile_options(-std=c++11)
add_executable(${PROJECT_NAME}_node src/merger.cpp)
target_link_libraries(${PROJECT_NAME}_node
    ${catkin_LIBRARIES}
)
```
and add the following:
```C++
find_package(catkin REQUIRED COMPONENTS
    roscpp
    rospy
    std_msgs
    image_transport
    std_msgs
    cv_bridge
)
```
Finally compile by catkin_make and test by running following command:
```bash
rosrun merger merger_node
```
Use the bash file start.sh to run the system.

#Camera Calibration Parameters for ORB-SLAM
In order to obtain features quickly, we need to make sure that the Camera Calibration parameters are set according to MINOS. Using the calibration method explained in our paper, we have calibrated MINOS front view camera, for Matterport3D indoor scenes (parameters provided below). These settings must be set in `/ORB_SLAM/Data/Settings.yaml`:
```yaml
%YAML:1.0

# Camera Parameters. Adjust them!

# Camera calibration parameters (OpenCV) 
Camera.fx: 890.246939
Camera.fy: 889.082597
Camera.cx: 378.899791
Camera.cy: 210.334985

# Camera distortion parameters (OpenCV) --
Camera.k1: 0.224181
Camera.k2: -1.149847
Camera.p1: 0.007295
Camera.p2: 0.0

# Camera frames per second 
Camera.fps: 10

# Color order of the images (0: BGR, 1: RGB. It is ignored if images are grayscale)
Camera.RGB: 1
```


## SLAM Recovery
The following files are included in this repository:
- `MINOS_Recovery_SLAM.py`: MINOS agent attempts to regain tracks using the trained network.
- `MINOS_Random_Movement.py`: MINOS agent attempts to regain tracks using random movements.
- `MINOS_Navigation.py`: MINOS agent does not attempt to regain tracks.

Weights Link: https://drive.google.com/file/d/1M9nby4k0wMmCy-EORqSqZ66UVNHwDcbn/view?usp=sharing

[![Demo Video](http://img.youtube.com/vi/Ru5zVv56EQk/0.jpg)](http://www.youtube.com/watch?v=Ru5zVv56EQk)
