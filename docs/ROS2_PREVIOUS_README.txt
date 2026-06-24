# D-CBF MPC Planner ROS2 Humble

A ROS2 Humble workspace port of the original ROS1 **MPC-D-CBF** planner.

- Original ROS1 project: <https://github.com/jianzhuozhuTHU/MPC-D-CBF>
- Paper: <https://arxiv.org/abs/2209.08539>
- Target platform: **Ubuntu 22.04 + ROS2 Humble + Gazebo Classic 11**

This repository preserves the paper/project pipeline: global path generation, local lidar map construction, obstacle ellipse fitting, Kalman prediction, and MPC with D-CBF obstacle constraints.

## What is included

```text
src/
  global_path_publisher/       Fixed straight-line /global_path publisher
  linear_path_publisher/       Goal-driven /global_path publisher from /goal_pose
  local_map/                   PointCloud2 -> grid map -> obstacle ellipses
  obs_param/                   Kalman prediction for obstacle ellipses
  local_planner/               CasADi/IPOPT MPC + D-CBF planner and controller
  scene/                       Gazebo/RViz launch files and moving cylinder helpers
  grid_map_cmake_helpers/      Vendored grid_map CMake helper package
  grid_map_rviz_plugin/        Vendored RViz2 grid map display plugin
  turtlebot3_velodyne_Gazebo/  Vendored TurtleBot3 + Velodyne Gazebo assets
```

The default full simulation uses `scene/worlds/world1.world`, which contains `cylinder_0` through `cylinder_4`. All five cylinders are configured to move through `scene/config/config.yaml`.

## Dependencies

Install ROS2 Humble first, then install common dependencies:

```bash
sudo apt update
sudo apt install -y \
  python3-colcon-common-extensions \
  python3-rosdep \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-rviz2 \
  ros-humble-pcl-ros \
  ros-humble-tf2-geometry-msgs \
  libeigen3-dev \
  libpcl-dev \
  libcgal-dev
```

Initialize rosdep if needed:

```bash
sudo rosdep init || true
rosdep update
```

Install package dependencies from the workspace root:

```bash
cd ~/DCBF_MPC_Planner_ROS2_GitHub
source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src --rosdistro humble -y
```

Install CasADi for the Python MPC solver:

```bash
python3 -m pip install --user casadi
```

## Build

```bash
cd ~/DCBF_MPC_Planner_ROS2_GitHub
source /opt/ros/humble/setup.bash
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo
source install/setup.bash
```

## Core smoke test without Gazebo

This starts the core planning graph without Gazebo/RViz. It is useful for checking the MPC/D-CBF nodes independently of simulator resources.

```bash
cd ~/DCBF_MPC_Planner_ROS2_GitHub
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch scene core_smoke.launch.py use_sim_time:=false launch_local_map:=false
```

With a synthetic predicted obstacle stream:

```bash
ros2 launch scene core_smoke.launch.py \
  use_sim_time:=false \
  launch_local_map:=false \
  publish_synthetic_obstacle:=true
```

Useful checks in another terminal:

```bash
source /opt/ros/humble/setup.bash
source ~/DCBF_MPC_Planner_ROS2_GitHub/install/setup.bash
ros2 topic list | grep -E '/global_path|/local_plan|/cmd_vel|/curr_state|/obs_predict_pub'
ros2 topic echo /cmd_vel --once
```

## Full Gazebo simulation

Start Gazebo, TurtleBot3 Velodyne, moving cylinders, local mapping, obstacle prediction, MPC/D-CBF planner, controller, and RViz:

```bash
cd ~/DCBF_MPC_Planner_ROS2_GitHub
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 launch scene sim.launch.py launch_local_planner:=true
```

Headless mode without Gazebo GUI or RViz:

```bash
ros2 launch scene sim.launch.py gui:=false rviz:=false launch_local_planner:=true
```

The robot does **not** move until a target is provided on `/goal_pose`.

In RViz, use **2D Goal Pose**. From a terminal, publish a simple forward goal:

```bash
ros2 topic pub --once /goal_pose geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: 'world'}, pose: {position: {x: 3.0, y: 0.0, z: 0.0}, orientation: {w: 1.0}}}"
```

Useful motion checks:

```bash
ros2 topic echo /global_path --once
ros2 topic echo /local_plan --once
ros2 topic echo /cmd_vel --once
ros2 topic echo /gazebo/model_states --once
```

### Notes

- Use `scene sim.launch.py` for the complete project. `scene start_backup.launch.py` is only a Gazebo world launcher and does not start the planner/controller or moving-cylinder helper.
- TurtleBot3's Gazebo diff-drive plugin already publishes `odom -> base_footprint`; keep `enable_odom_tf` at its default `false` unless you intentionally replace that TF path.
- `local_planner/config/config.yaml` uses `safe_dist: 0.5` for TurtleBot3/Gazebo feasibility. If you increase it, verify that IPOPT still finds feasible solutions and that obstacle clearance remains acceptable.
- `scene/config/config.yaml` sets `cylinder_num: 5`, so `cylinder_0` through `cylinder_4` move in the default `world1.world` scene.

## Main topic contract

| Topic | Message | Producer | Consumer |
| --- | --- | --- | --- |
| `/goal_pose` | `geometry_msgs/msg/PoseStamped` | RViz / user | `linear_path_publisher` |
| `/global_path` | `nav_msgs/msg/Path` | `global_path_publisher` or `linear_path_publisher` | `local_planner` |
| `/velodyne_points` | `sensor_msgs/msg/PointCloud2` | Gazebo Velodyne | `local_map` |
| `/for_obs_track` | `std_msgs/msg/Float32MultiArray` | `local_map` | `obs_param` |
| `/obs_predict_pub` | `std_msgs/msg/Float32MultiArray` | `obs_param` | `local_planner` |
| `/curr_state` | `std_msgs/msg/Float32MultiArray` | `controller` | `local_planner` |
| `/local_plan` | `std_msgs/msg/Float32MultiArray` | `local_planner` | `controller` |
| `/cmd_vel` | `geometry_msgs/msg/Twist` | `controller` | TurtleBot3 diff-drive plugin |
| `/cmd_move` | `std_msgs/msg/Bool` | `local_planner` | moving cylinder helper |

## Verification performed before packaging

The following checks passed on the development machine:

```bash
source /opt/ros/humble/setup.bash
colcon build --symlink-install
colcon test --packages-select linear_path_publisher local_planner scene
colcon test-result --verbose
```

Runtime checks performed:

- Core smoke launch with and without synthetic obstacle input.
- Full headless Gazebo launch.
- Published `/goal_pose` at `(3.0, 0.0)` and observed TurtleBot3 motion to approximately `(2.9999, -0.0013)`.
- Verified `/global_path` ended at `(3.0, 0.0)`.
- Verified `/local_plan` and `/cmd_vel` became non-zero.
- Verified all five default cylinders moved in `/gazebo/model_states`.

## Uploading to GitHub

This directory is intentionally packaged without `build/`, `install/`, `log/`, `.git/`, cache files, or generated Python bytecode.

To upload as a new repository:

```bash
cd ~/DCBF_MPC_Planner_ROS2_GitHub
git init
git add .
git commit -m "Initial ROS2 Humble D-CBF MPC planner port"
git branch -M main
git remote add origin <YOUR_GITHUB_REPO_URL>
git push -u origin main
```

Replace `<YOUR_GITHUB_REPO_URL>` with your GitHub repository URL.

## License / attribution

This is a ROS2 Humble port derived from the upstream MPC-D-CBF ROS1 project and uses vendored third-party ROS/Gazebo packages. Before publishing publicly, review the upstream repositories' licenses and retain their copyright/license notices.
