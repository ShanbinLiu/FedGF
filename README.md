# FedGF

一个用于**联邦学习（Federated Learning）算法研究与对比实验**的 PyTorch 项目，支持多种经典/改进聚合方法，并提供统一的训练入口、客户端评估、Gini 公平性指标统计与结果落盘流程。

> 当前代码主入口为 `main.py`，通过 `--method`、`--dataset`、`--model` 等参数动态加载对应模块。

---

## 1. 项目特点

- 统一训练入口：`main.py` 负责读取参数、初始化数据/模型/方法并执行联邦训练。
- 方法模块化：各联邦方法位于 `method/`，均复用 `BaseServer/BaseClient` 框架。
- 任务模块化：每个数据集任务位于 `task/<dataset>/`，包含模型定义与数据目录。
- 多指标记录：训练中输出 `train_acc / train_loss / test_acc / test_std / test_gini` 等指标。
- 公平性分析：默认统计各客户端精度的 Gini 系数（衡量分布不均衡程度）。

---

## 2. 目录结构

```text
FedGF/
├── main.py                  # 训练入口
├── log.py                   # 日志输出工具
├── utils/
│   └── tools.py             # 参数解析、随机种子、初始化、输出命名
├── method/
│   ├── fedbase.py           # BaseServer/BaseClient
│   ├── fedavg.py            # FedAvg
│   ├── fedprox.py           # FedProx
│   ├── qfedavg.py           # q-FedAvg
│   ├── afl.py               # AFL
│   ├── fedfa.py             # FedFA
│   ├── fedfv*.py            # FedFV 系列
│   ├── fedmgda+.py          # FedMGDA+
│   ├── FedGini.py           # FedGini
│   └── fedgf*.py            # FedGF
└── task/
    ├── datafuncs.py         # 数据集封装
    ├── modelfuncs.py        # 训练/测试与模型字典操作
    ├── cifar01/
    ├── office10/
    ├── office10_resnet18/
    └── synthetic_*/         # 合成数据任务与生成脚本
```

---

## 3. 环境要求

建议使用 Python 3.8+。

核心依赖（按代码导入整理）：

- `torch`, `torchvision`
- `numpy`
- `tqdm`
- `cvxopt`（`fedmgda+` 需要）
- `opencv-python`, `Pillow`
- `SimpleITK`, `nibabel`（`office10*` 数据处理脚本中使用）

可先安装基础依赖：

```bash
pip install torch torchvision numpy tqdm cvxopt opencv-python pillow SimpleITK nibabel
```

---

## 4. 数据准备

初始化逻辑会默认读取以下目录（每个目录下可有多个 `.json` 文件）：

- `task/<dataset>/data/train`
- `task/<dataset>/data/vaild`
- `task/<dataset>/data/test`

每个 JSON 文件应包含 `user_data` 字段，并按客户端名称组织数据。

> 注意：代码中验证集目录拼写为 `vaild`（不是 `valid`），准备数据时请保持一致。

合成数据可通过对应脚本生成，例如：

```bash
cd task/synthetic_1_1
python generate_synthetic.py
```

---

## 5. 快速开始

在仓库根目录执行：

```bash
python main.py \
  --method fedavg \
  --dataset cifar01 \
  --model resnet18 \
  --num_rounds 100 \
  --num_epochs 1 \
  --batch_size 64 \
  --learning_rate 0.1 \
  --proportion 0.1 \
  --gpu 0
```

运行流程：

1. 读取命令行参数并设置随机种子。
2. 动态加载 `task.<dataset>.<model>` 中的 `Model/Loss`。
3. 加载客户端数据并构建客户端对象。
4. 动态加载 `method.<method>` 中的 `Server/Client`。
5. 进入联邦训练与周期评估。
6. 将结果保存到 `task/<dataset>/record/`。

---

## 6. 常用参数说明

### 通用参数

- `--method`：联邦方法名（如 `fedavg`、`fedprox`、`fedgf`）
- `--dataset`：任务名（对应 `task/<dataset>/`）
- `--model`：模型模块名（对应 `task/<dataset>/<model>.py`）
- `--num_rounds`：通信轮数
- `--proportion`：每轮采样客户端比例
- `--num_epochs`：客户端本地训练 epoch 数
- `--learning_rate`：本地学习率
- `--batch_size`：本地批大小
- `--optimizer`：`SGD` 或 `Adam`
- `--sample`：客户端采样策略（`uniform`/`prob`）
- `--aggregate`：服务器聚合策略（`uniform`/`weighted_scale`/`weighted_com`）
- `--gpu`：GPU 编号，`-1` 为 CPU

### 方法特有参数（按实现）

- `afl`：`--learning_rate_lambda`
- `qfedavg`：`--q`
- `fedmgda+`：`--epsilon`
- `fedprox`：`--mu`
- `fedfa`：`--beta`, `--gamma`, `--momentum`
- `fedfv/fedfv_random/fedfv_reverse`：`--alpha`, `--tau`
- `FedGini`：`--epsilons`, `--window`
- `fedgf/fedgf_cifar`：`--lamda`, `--threshold`, `--eta`

---

## 7. 输出结果

训练完成后会在 `task/<dataset>/record/` 生成 JSON 记录文件，文件名会包含方法参数与训练超参。

主要字段包括：

- `train_acc`
- `train_loss`
- `test_acc`
- `test_std`
- `test_gini`
- `client_accs`
- `best_test_gini`

此外，服务器会将验证最优轮次对应模型保存为：

- `task/<dataset>/record/best_model.pth`

---

## 8. 已实现方法

当前 `method/` 下可见实现（不含基类/工具模块）：

- FedAvg
- FedProx
- q-FedAvg
- AFL
- FedFA
- FedFV / FedFV-random / FedFV-reverse
- FedMGDA+
- FedGini
- FedGF（含 `fedgf_cifar` 变体）

---

## 9. 常见问题

### Q1: 报错找不到 `task/<dataset>/data/...` 目录？
请先按第 4 节准备数据目录与 JSON 文件，且目录名需包含 `vaild`。

### Q2: 报错 `No module named task.<dataset>.<model>`？
请确认：

- `task/<dataset>/` 下存在 `<model>.py`
- 文件中定义了 `Model` 与 `Loss` 类

### Q3: 结果文件没有生成？
请确认 `task/<dataset>/record/` 目录存在且具有写权限。

---

## 10. 许可证与致谢

本仓库当前未提供独立许可证文件。若你计划公开发布或二次分发，建议补充 `LICENSE` 并明确数据集/算法引用来源。
