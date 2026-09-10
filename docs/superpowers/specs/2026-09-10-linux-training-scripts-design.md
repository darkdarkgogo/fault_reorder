# Linux 安装与训练脚本设计

## 目标

为无法联网的 Linux 训练主机提供两个 Bash 入口。脚本使用用户当前激活环境中的
Python，不访问网络、不创建 Conda 环境、不调用 sudo，也不替用户选择
CUDA/PyTorch 安装源。训练所需的 16 组 fault embedding 和 pybind11 头文件随
仓库提交。

## 脚本

`scripts/setup_linux.sh` 从任意工作目录定位仓库根目录，检查 Python 3.9+、
PyTorch、NumPy、setuptools、Python 开发头和 C++ 编译器，使用仓库内的 pybind11
头文件编译当前平台的 `cpp_podem` 扩展，验证扩展可导入，最后对指定 manifest
执行数据校验。脚本不执行 pip。manifest 默认是 `configs/all_benchmarks.json`，
可用第一个位置参数覆盖。

`scripts/run_linux.sh` 从任意工作目录定位仓库根目录，并提供 `validate`、`smoke`、
`train`、`resume` 和 `evaluate` 子命令。`smoke` 默认跑 1 轮两个电路；`train`
默认跑 100 轮全部 16 个电路；恢复参数中的轮数表示训练后的总目标轮数。脚本用
`exec` 启动训练程序，以便终端信号直接交给 Python 并由现有 checkpoint 逻辑处理。

两个脚本默认调用当前 `PATH` 中的 `python3`，也允许通过 `PYTHON_BIN` 指定同一
环境中的其他解释器路径。包含 `/` 的相对解释器路径会在切换到仓库根目录前转换
为绝对路径。参数错误、缺少解释器、编译失败、模块导入失败和数据校验失败都立即
返回非零状态。

## 编译产物

仓库不再保存平台和 Python ABI 相关的 `PODEM/*.pyd` 与 `PODEM/build/`。
`.gitignore` 同时忽略 Windows `.pyd`、Linux `.so` 和构建目录。Linux 用户运行
安装脚本后，在本机生成与当前解释器匹配的 `.so`。

`training_data/fault-order-embeddings` 保存训练校验实际使用的 embedding NPZ 和
精简 metadata JSON。门级 embedding 与完整图 JSON 不参与训练，因此不随仓库
分发。manifest 直接引用这些离线文件，新克隆不执行 DeepGate2 推理。

## 验证

对两个脚本执行 Bash 语法检查，检查帮助和参数路由，并运行 Python CLI 回归测试。
Linux 上的最终验收命令为 `./scripts/setup_linux.sh` 和
`./scripts/run_linux.sh smoke`；完整训练由用户在 Linux 上显式启动。
