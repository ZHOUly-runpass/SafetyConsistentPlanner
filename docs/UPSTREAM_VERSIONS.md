# Upstream Repository Versions

## PaIR-Drive

Repository:
https://github.com/zhexilian/PaIR-Drive.git

Local path:
../01_PaIR-Drive_Upstream

Commit:
d6aa92b2c8d0136a74f793199146b0593fe33869

Role:
Learning-based trajectory-planning reference.

Modification policy:
Read-only upstream repository.


## ROS2 D-CBF-MPC

Repository:
https://github.com/YSU-NavPilotHub/Ros2_D-CBF-MPC.git

Local path:
../02_Ros2_D-CBF-MPC_Baseline

Commit:
275007927510650709f67bcfc6f4f5aec6f8a1dd

Role:
Reproduced ROS2 D-CBF-MPC baseline, offline safety teacher,
and online safety execution backend.

Modification policy:
Keep the baseline repository clean.
All interfaces, bridges, patches, and new project code belong
to 03_SafetyConsistentPlanner_Workshop.
