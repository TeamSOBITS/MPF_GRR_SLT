[JA](README.md) | [EN](README.en.md)

# MPF_GRR_SLT

このパッケージは，`sobit_follower` パッケージの対象者識別機能として使用されます．
そのため，使用するのは `mono_following` パッケージのみです．

`mono_following` パッケージでは，元のパッケージで使用されていた人物特徴抽出器の代わりに，OSNet と呼ばれる人物特徴抽出器を使用しています．
人物特徴抽出器以外の仕様は，基本的に元のパッケージから変更していません．
OSNet の詳細については，下記の Related packages を参照してください．

## Dependencies

使用したコンピュータ環境:

* Ubuntu 18.04 または 20.04
* Melodic または Noetic
* GTX 2060 / GTX 1650

1. conda 環境を作成し，PyTorch をインストールします．
   Docker 環境内で使用する場合，この設定は不要です．

```bash
conda create -n mono_following python=3.8
conda activate mono_following
# GPU環境に合わせた設定です．他の環境では設定に注意してください
conda install pytorch torchvision cudatoolkit=10.2 -c pytorch
```

2. Python 関連パッケージをインストールします．

```bash
pip install -r requirements.txt
cd ros_numpy
python setup.py install
```

3. C++ 関連パッケージをインストールします．

* OpenCV==3.4
* Eigen==3.0+

## Our tf tree

```bash
base_link->camera_link->camera_optical_link
```

## How to use

### Tracking

```bash
# 横幅ベースの単眼人物追跡を起動
# rosbagで実行する場合は use_sim_time:=true を指定します
# 画像トピックがcompressedの場合は sim:=true を指定します
roslaunch mono_tracking all_mono_tracking.launch sim:=true use_sim_time:=true

# 実機ロボットで実行する場合は use_sim_time:=false を指定します
roslaunch mono_tracking all_mono_tracking.launch sim:=false use_sim_time:=false
```

* Input: `/camera/color/image_raw`
* Output: `mono_tracking/msg/TrackArray.msg`

### Target Identification

```bash
# GRR_SLT_MPFによる人物追従を起動します
# rosbagで実行する場合は use_sim_time:=true を指定します
roslaunch mono_followng mono_following.launch use_sim_time:=true

# 実機ロボットで実行する場合は use_sim_time:=false を指定します
roslaunch mono_following mono_following.launch use_sim_time:=false
```

* Input: `mono_tracking/msg/TrackArray.msg`
* Output: `mono_following/msg/Target.msg`

### Controlling and Following

```bash
# 制御ノードを起動します
roslaunch mono_control mono_controlling.launch
```

* Input: `mono_following/msg/Target.msg`; `/bluetooth_teleop/joy`
* Output: `/cmd_vel`

## Target identification Test

<img src="pictures/performance.gif" alt="performance" style="zoom: 100%;" />

## TODO

* コードの簡潔性を改善する
* 評価結果を公開する

## Citation

## Related packages

* [sobit_follower](https://github.com/TeamSOBITS/sobit_follower)
* [deep-person-reid(OSNet)](https://github.com/KaiyangZhou/deep-person-reid)

## Acknowledge

* [YOLOX_deepsort_tracker](https://github.com/pmj110119/YOLOX_deepsort_tracker)
* [monocular_person_following](https://github.com/koide3/monocular_person_following)

## Paper

* Hanjing Ye，Jieting Zhao，Yaling Pan，Weinan Chen and Hong Zhang，“Following Closely: A Robust Monocular Person Following System for Mobile Robot”，arXiv preprint arXiv:2204.10540，2022 [[link]](https://arxiv.org/abs/2204.10540)．
