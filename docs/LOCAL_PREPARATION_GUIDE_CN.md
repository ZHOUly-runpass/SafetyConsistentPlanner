# 本地代码准备指南

## 概述

本文档说明在本地 Windows 机器上完成代码准备后，如何打包并迁移到 Linux 开发机。

## 本地准备内容

### 已完成的文件

| 类别 | 文件 | 说明 |
|------|------|------|
| 接口 | `src/safety_planner/interfaces/` | 10个数据接口定义，含 ranking、shape 检查和 schema_version |
| 车辆模型 | `src/safety_planner/vehicle/` | 运动学自行车模型、坐标转换、几何工具 |
| D-CBF-MPC | `src/safety_planner/dcbf_mpc/` | barrier、CBF residual、重采样、诊断、NumPy reference solver、CasADi backend 入口 |
| 规划执行 | `src/safety_planner/planning/` | 候选过滤、排序和 fallback 执行 |
| 安全教师 | `src/safety_planner/safety_teacher/` | MpcResult 到伪标签记录的转换 |
| 评估 | `src/safety_planner/evaluation/` | ADE、FDE、yaw、velocity 基础指标 |
| 模型框架 | `src/safety_planner/models/` | 场景编码器、IL头、B-spline头、安全头、排序器 |
| 训练框架 | `src/safety_planner/training/` | 损失函数、训练脚本入口 |
| 配置 | `configs/` | 10个YAML配置文件 |
| 测试 | `tests/unit/` | 10个测试文件，40个本地测试用例 |
| 文档 | `docs/` | 项目范围、接口规格、ROS2参考映射 |

### 本地已跳过（需要服务器完成）

- 严格 CasADi/IPOPT 非线性优化求解器（本地已有 NumPy reference backend）
- 真实 nuPlan 数据加载（本地支持 JSON fixture，不自动下载数据）
- GPU 训练流程和大规模训练结果
- GPU训练、闭环仿真

## 迁移到开发机

### 步骤

1. 将整个 `03_SafetyConsistentPlanner_Workshop` 目录打包：
   ```bash
   # Windows PowerShell
   Compress-Archive -Path "D:\E2Eproject_MPC\03_SafetyConsistentPlanner_Workshop\*" -DestinationPath "workshop.zip"
   ```

2. 上传到开发机并解压：
   ```bash
   scp workshop.zip user@dev-machine:/home/user/
   ssh user@dev-machine
   cd /home/user/
   unzip workshop.zip -d 03_SafetyConsistentPlanner_Workshop
   ```

3. 验证文件完整性：
   ```bash
   cd 03_SafetyConsistentPlanner_Workshop
   python -m pytest tests/unit/ -v
   ```

4. 设置环境（在开发机上执行）：
   ```bash
   conda create -n safety_planner python=3.10 -y
   conda activate safety_planner
   pip install -r environment/learning/requirements.txt
   pip install -r environment/solver/requirements.txt
   ```

### 注意事项

- 本地代码不依赖 nuPlan、ROS2、CUDA
- 所有硬编码路径在 `configs/paths/server.example.yaml` 中需修改为实际路径
- solver 配置中的 IPOPT 参数可在 `configs/mpc/solver.yaml` 中调整
- 车辆参数最终以 nuPlan 配置为准，`configs/mpc/vehicle_bicycle.yaml` 为建议初始值
