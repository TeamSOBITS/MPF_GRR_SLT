# MPF_GRR_SLT

This package is used as the target identification function of the sobit_follower package．For this reason，only the mono_following package is used．
In the mono_following package，the human feature extractor called OSNet is used instead of the human feature extractor used in the original package．
(Other than the human feature extractor，the other specifications are basically unchanged．)For details of OSNet，please refer to the Related packages below．

## Dependencies
Our computer settings:
- Ubuntu 18.04 or 20.04
- Melodic or Noetic
- GTX 2060/ GTX 1650
1. Create a conda environment and install pytorch(When used within a Docker environment, these settings are not required．)
```bash
conda create -n mono_following python=3.8
conda activate mono_following
# This is based on your GPU settings, other settings should be careful
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```
2. Install python related packages:
```bash
pip install -r requirements.txt
cd ros_numpy
python setup.py install
```
3. Install cpp related packages:

- OpenCV==3.4
- Eigen==3.0+

## Our tf tree
```bash
base_link->camera_link->camera_optical_link
```

## How to use

### Tracking

```bash
# launch width-based monocular people tracking
# If running with rosbag, use_sim_time:=true; if the image topic is compressed, sim:=true
roslaunch mono_tracking all_mono_tracking.launch sim:=true use_sim_time:=true
# If running in real robot, use_sim_time:=false;
roslaunch mono_tracking all_mono_tracking.launch sim:=false use_sim_time:=false
```
- Input: /camera/color/image_raw
- Output: mono_tracking/msg/TrackArray.msg

### Target Identification

```bash
# launch our GRR_SLT_MPF person following, use_sim_time:=true for rosbag
roslaunch mono_followng mono_following.launch use_sim_time:=true
# launch our GRR_SLT_MPF person following, use_sim_time:=false for robot running
roslaunch mono_following mono_following.launch use_sim_time:=false
```
- Input: mono_tracking/msg/TrackArray.msg
- Output: mono_following/msg/Target.msg


### Controlling and Following

```bash
# launch control
roslaunch mono_control mono_controlling.launch
```
- Input: mono_following/msg/Target.msg; /bluetooth_teleop/joy
- Output: /cmd_vel

## Target identification Test
<img src="pictures/performance.gif" alt="performance" style="zoom: 100%;" />

## TODO
- Improve simplicity of the code
- Release evaluation results

## Citation
## Related packages
- [sobit_follower](https://github.com/TeamSOBITS/sobit_follower)
- [deep-person-reid(OSNet)](https://github.com/KaiyangZhou/deep-person-reid)

## Acknowledge
- [YOLOX_deepsort_tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker)
- [monocular_person_following](https://github.com/koide3/monocular_person_following)

## Paper
- Hanjing Ye，Jieting Zhao，Yaling Pan，Weinan Chen and Hong Zhang，“Following Closely: A Robust Monocular Person Following System for Mobile Robot”，arXiv preprint arXiv:2204.10540，2022 [[link]](https://arxiv.org/abs/2204.10540)．


