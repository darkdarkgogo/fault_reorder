# Linux 安装与训练脚本设计

## 目标

为无法联网的 Linux 训练主机提供两个 Bash 入口。脚本使用用户当前激活环境中的
Python，不访问网络、不创建 Conda 环境、不调用 sudo，也不替用户选择
CUDA/PyTorch 安装源。训练所需的 16 组 fault embedding 和 pybind11 头文件随
仓库提交。

## 脚本

`scripts/setup_linux.sh` 从任意工作目录定位仓库根目录，然后只执行
`python3 PODEM/setup.py build_ext --inplace`。构建器使用仓库内的 pybind11 头文件；
脚本不执行环境探测、pip、联网或数据生成。数据校验由训练入口完成，因此错误
直接来自实际失败的组件。

`scripts/run_linux.sh` 从任意工作目录定位仓库根目录并直接训练全部 16 个电路。
第一个可选参数是轮数，默认 100；第二个可选参数是输出目录，默认
`runs/shared_scorer`。脚本用 `exec` 启动 Python，使终端信号直接交给训练程序。

两个脚本固定调用当前 `PATH` 中的 `python3`。编译或训练失败时直接返回 Python
命令的原始错误和非零状态。

## 编译产物

仓库不再保存平台和 Python ABI 相关的 `PODEM/*.pyd` 与 `PODEM/build/`。
`.gitignore` 同时忽略 Windows `.pyd`、Linux `.so` 和构建目录。Linux 用户运行
安装脚本后，在本机生成与当前解释器匹配的 `.so`。

`training_data/fault-order-embeddings` 保存训练校验实际使用的 embedding NPZ 和
精简 metadata JSON。门级 embedding 与完整图 JSON 不参与训练，因此不随仓库
分发。manifest 直接引用这些离线文件，新克隆不执行 DeepGate2 推理。

## 验证

对两个脚本执行 Bash 语法检查，检查帮助和参数路由，并运行 Python CLI 回归测试。
Linux 上的最终命令为 `./scripts/setup_linux.sh` 和 `./scripts/run_linux.sh`。
