# Jetson Orin Nano Super 边缘 AI 学习路线（8GB + IMX219）

> **适用设备**：Jetson Orin Nano Super 8GB，IMX219 CSI 摄像头，系统已配置完成
> **节奏参考**：每天 1 小时 ≈ 6 个月走完全程；每天 2 小时 ≈ 3 个月；只有周末 ≈ 8~10 个月
> **唯一铁律**：**每个阶段必须产出一个"能给别人看的东西"**（一张表 / 一个仓库 / 一段视频）。没有产出的学习，在这个领域等于没学。

---

## 0. 先看清地图：边缘 AI 到底包含什么

```
应用层    你的项目：视觉检测 / 语音助手 / 多模态问答 / 机器人
编排层    Docker、GStreamer 流水线、systemd 服务、ROS 2
推理层    TensorRT、TensorRT-Edge-LLM、DeepStream、llama.cpp、Ollama
加速层    CUDA、cuDNN、VPI、NVDEC / NVENC（硬件编解码）
系统层    JetPack / L4T / Ubuntu、驱动、功耗与散热
硬件层    Orin Nano Super：GPU + 8GB 统一内存 + CSI 摄像头 + NVMe
```

**一句话定义你的目标能力**：

> 边缘 AI 工程师的价值 = **把模型塞进有限的内存和功耗里，还能跑得足够快、足够稳。**

所以这条路线的重点**不是"训练模型"**，而是**"部署、量化、加速、稳定运行"**。这四件事在 PC 和云端都很难真正学到，恰好是你这块板子最擅长教的。

---

## 1. 能力分级：怎么判断"我学到什么地步了"

| 等级 | 特征 | 你应该能回答的问题 |
|---|---|---|
| **L0 看得懂** | 知道名词：推理、量化、TensorRT、GStreamer | "TensorRT 是干什么的？" |
| **L1 跑得通** | 能照教程把 demo 跑起来 | "我能独立复现官方示例吗？" |
| **L2 调得动** | 能换模型、换参数、看指标 | "模型跑不动时，我先看内存还是先看散热？" |
| **L3 改得了** | 能改代码、换流水线、自己做转换与量化 | "我能把 PyTorch 模型转成 TensorRT 并测出加速比吗？" |
| **L4 做得出** | 能设计一个连续跑 24 小时不崩的完整服务 | "断电重启后它能自己恢复吗？" |

本路线四个阶段结束后，目标是把你从 **L0 送到 L4**。

---

## 2. 四个阶段（每个阶段都有明确的"过关测试"）

### 阶段 0 ｜ 把板子变成一台真正的开发机（第 1 周）

**目标**：脱离显示器，从自己的电脑上舒服地开发这块板子，并掌握它的"身体状况"。

**学什么**

1. **确认环境版本**（一切教程的前提）
   ```bash
   cat /etc/nv_tegra_release      # 看 L4T 版本
   uname -a                       # 内核
   ```
   ⚠️ 记住：JetPack 6.x 和 7.x 的教程**不通用**，抄教程前先对版本。
2. **远程开发流（本阶段最重要的技能）**
   - 板子开 SSH，电脑用 VS Code / Cursor 的 Remote-SSH 直接编辑板子上的文件
   - 功能：代码补全、终端、传文件，全部在板子上跑 —— 以后你不用再插显示器
3. **存储与容器**
   - 装 NVMe SSD 并把系统/模型/Docker 数据放在 SSD（官方有专门教程）
   - 了解 `jetson-containers`：NVIDIA 官方的 Jetson 容器工具，后面所有教程都靠它
4. **性能与功耗基线**
   ```bash
   sudo nvpmodel -q          # 当前功耗模式（目标是 MAXN SUPER）
   tegrastats                # 实时看内存/温度/功耗/GPU 占用（最重要的一个命令）
   sudo jetson_clocks        # 锁最高频率
   ```
5. **内存腾挪（为后面跑大模型做准备）**
   - 远程开发时关掉桌面 GUI，可省约 **800MB**：`sudo systemctl set-default multi-user.target`
   - swap 放在 NVMe 上，避免大模型加载时直接 OOM
   - ⚠️ **如果你要用 CSI 摄像头，不要禁用 `nvargus-daemon` 服务**（网上一些"省内存"教程会让你关它，关了就拍不了照）

**交付物**：一页《板子体检报告》——系统版本、固件版本、存储、空闲内存、满载温度、功耗模式。

**过关测试**
- [ ] 我能只用自己的电脑（不接显示器）完成：写代码 → 在板子上运行 → 看输出
- [ ] 我能在 10 秒内用 `tegrastats` 说出当前内存占用和温度

---

### 阶段 1 ｜ 会"跑"模型：从零到第一个 AI 输出（第 2~4 周）

**目标**：能独立把别人做好的模型在你的板子上跑起来，并说清它跑得好不好。

**学什么**

1. **必须搞懂的概念（用大白话，别背术语）**
   - 模型 / 权重 / 参数量：模型就是一堆数字，"3B" 指 30 亿个参数
   - **推理 ≠ 训练**：你只做推理（用模型），不训练（造模型）
   - **延迟 vs 吞吐**：延迟 = 一次要多久；吞吐 = 每秒能处理几个
   - **量化**：把 FP32 换成 FP16 / INT8 / INT4，模型更小更快，精度略降 —— 这是边缘 AI 的核心手段
   - **统一内存**：CPU 和 GPU 共用 8GB，所以"模型占的内存"和"系统占的内存"是抢同一块。这是理解一切 8GB 问题的钥匙
2. **跑通官方现成示例**：用 [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/) + `jetson-containers`，跑分类、目标检测、分割模型（一条命令级别，先不用懂原理）
3. **让大模型开口说话**：按官方 [Ollama on Jetson](https://www.jetson-ai-lab.com/tutorials/ollama/) 一键安装，跑一个 3B 级别的模型（Ollama 已原生支持 Jetson CUDA；**强烈建议模型放 NVMe**，容器约 7GB + 模型 5GB 以上）
4. **测量三件事**：FPS / 单次延迟 / 内存占用。养成"跑完必测"的习惯

**交付物**：一张《模型性能对比表》—— 至少 3 个模型 × (FPS、延迟、内存、温度)。

**过关测试**
- [ ] 我能不看教程，独立把一个新的模型在板子上跑起来
- [ ] 模型报内存错误（OOM）时，我知道先做什么（看 `tegrastats` → 关程序 → 换更小/更低的量化）
- [ ] 我能解释"为什么这个模型在板子上比在电脑上慢"

---

### 阶段 2 ｜ 视觉主线 + TensorRT 真功夫（第 5~10 周）⭐ 本路线最核心的一段

**目标**：把"能跑"变成"跑得快"，掌握边缘 AI 最值钱的技能：**模型转换与量化加速**。

**学什么**

1. **CSI 摄像头流水线**
   - 你的 IMX219 是 CSI 摄像头（走 ISP，不占 CPU），和 USB 摄像头完全是两条路
   - 学 GStreamer：`nvarguscamerasrc` → `nvvidconv` → `nv3dsink`（本机预览）/ 推流
   - 理解 **NVMM 零拷贝**：数据不进 CPU 内存，直接在 GPU 侧流转 —— 这是 Jetson 快的原因
   - 排障：`ls /dev/video*`、`dmesg | grep -i imx219`、排线方向
2. **PyTorch → ONNX → TensorRT 完整链路**
   - 导出 ONNX、用 `trtexec` 构建 engine、跑 FP16 / INT8
   - 会用量化校准做 INT8，知道精度掉了多少
   - 会看 TensorRT 的日志和 profile 输出
3. **OpenCV（CUDA 版）**：图像预处理不要用 CPU 卡住流水线
4. **DeepStream 入门**：Jetson 的招牌框架，把"摄像头 → 解码 → 推理 → 跟踪 → 输出"做成标准流水线；从单路做到多路
5. **性能剖析**：Nsight Systems 看时间花在哪（解码？预处理？推理？后处理？）

**交付物**：**IMX219 实时目标检测 demo**——CSI 摄像头输入、TensorRT INT8 推理、浏览器或 RTSP 看结果；附一页报告：FP32 vs FP16 vs INT8 的 FPS 与精度对比。

**过关测试**
- [ ] 我能把一个自己选的 PyTorch 模型转成 TensorRT engine 并跑起来
- [ ] 我能说清 INT8 相比 FP32 快了多少、精度掉了多少
- [ ] 我的摄像头流水线从采集到显示的总延迟我能测出来并解释组成

---

### 阶段 3 ｜ 端侧大模型 / 多模态 / Agent（第 11~16 周）

**目标**：在 8GB 上把"生成式 AI"真正跑起来，并学会量化与显存调度的取舍。

**学什么**

1. **端侧 LLM 三级递进**
   - 入门：Ollama（简单，适合验证）
   - 进阶：llama.cpp / 容器化推理（能调参数、能换量化格式）
   - 硬核（推荐重点学）：[TensorRT Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) —— 英伟达官方的 **C++ 推理运行时**，推理路径里没有 Python；官方教程就是用 **Qwen3-4B-Instruct 在 Orin Nano 上**的例子，讲 INT4 / NVFP4 / FP8 量化、ONNX 导出、engine 构建。这是最能体现"边缘部署"深度的一环
2. **模型压缩工具链**：ModelOpt 量化、（可选）投机解码加速
3. **多模态（VLM）**：参考官方 [live-vlm-webui](https://www.jetson-ai-lab.com/) 做"摄像头画面 + 提问"；注意 8B 级 VLM 官方示例是跑在 Thor 上的，**Orin Nano 上要选更小的 VLM，或退化成"检测模型 + LLM"两步方案**
4. **语音**：Whisper 做识别 + TTS 做播报 → 串成一个语音助手（难点是三个模型不能同时驻留，要分时加载）
5. **RAG 本地知识库**：embedding + 向量库 + 本地 LLM，做一个离线的"问我的资料"
6. **Agent 方向（2026 年的新趋势，值得关注）**：Jetson 上的 agent skills 与多模态 AI 工作流（官方教程里已有对应内容），理解"模型 + 工具调用"如何在端侧落地

**交付物**：一个**离线多模态问答**或**语音助手** demo，附"量化前后对比"（模型体积、内存、速度、效果）。

**过关测试**
- [ ] 拔掉网线，它也能完成"看图/听声 → 给出回答"
- [ ] 我能解释：为什么这个模型必须用 INT4 才能装进 8GB
- [ ] 我能针对"同时要跑 ASR + LLM + TTS"设计一个可行的时间/内存调度方案

---

### 阶段 4 ｜ 工程化 + 综合项目（第 17~24 周）

**目标**：从"能跑"到"能交付"。这一步决定你是"玩过 demo"还是"做过项目"。

**学什么**

1. **工程化**
   - 用 Docker 把整个应用封起来（换台板子也能跑）
   - 用 systemd 做开机自启、崩溃自动重启
   - 日志与异常处理：摄像头掉线、网络断开、内存不足怎么办
   - 功耗与散热管理：长期运行下的温度曲线、降频策略
2. **选一个主方向深挖**（三选一）
   - **A. 视觉/视频分析**：DeepStream 多路视频、跟踪、跨线计数、安防 NVR
   - **B. 生成式 AI 助手**：多模态 + 语音 + RAG 的家庭助手
   - **C. 机器人**：ROS 2 / Isaac ROS，把摄像头 + 模型接到运动控制上
3. **稳定性验证**：连续运行 24~72 小时，记录内存/温度是否漂移

**交付物**：一个 **GitHub 仓库**（README 让别人能一步步复现）+ 性能报告 + 一段演示视频。

**过关测试**
- [ ] 别人拿着我的 README 能复现我的项目
- [ ] 断电重新上电后，服务自动恢复并正常工作
- [ ] 连续跑 24 小时不崩、内存不持续增长、温度稳定

---

## 3. 8GB 的七条生存法则（全程适用）

1. **一切先看 `tegrastats`**：内存、温度、功耗、GPU 占用，所有问题的第一现场。
2. **能量化就量化**：INT8 / INT4 优先，精度一般掉很少、速度提升明显。
3. **模型和容器镜像全放 NVMe**：SD 卡会严重拖慢加载。
4. **远程开发时关桌面 GUI**：白捡约 800MB 内存。
5. **优先小模型**：LLM 选 1.5B~4B；检测模型选 nano / small 级别。别跟 8GB 硬碰。
6. **模型分时加载，不要全驻留**：ASR + LLM + TTS 同时待在内存里必然 OOM，要做"用完就卸"。
7. **swap 放 NVMe**：它是保命手段，不是提速手段，跑起来会很慢。

---

## 4. 资源清单（按使用顺序）

| 用途 | 资源 |
|---|---|
| 官方硬件/系统文档 | [Jetson Orin Nano Developer Kit User Guide](https://docs.nvidia.com/jetson/orin-nano-devkit/user-guide/latest/) |
| 上手 + 远程开发 + 容器 + 内存优化 + 各模型教程 | [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com/)（**本路线的"主线教材"**） |
| 容器与教程源码 | [NVIDIA-AI-IOT/jetson-ai-lab](https://github.com/NVIDIA-AI-IOT/jetson-ai-lab)、[jetson-containers](https://github.com/dusty-nv/jetson-containers) |
| 端侧 LLM/VLM 高性能推理 | [TensorRT-Edge-LLM](https://github.com/NVIDIA/TensorRT-Edge-LLM) |
| 视频 AI 框架 | [DeepStream 官方文档](https://docs.nvidia.com/metropolis/deepstream/dev-guide/) |
| 机器人 | [Isaac ROS 文档](https://nvidia-isaac-ros.github.io/) |
| 报错求助 | [NVIDIA Jetson 论坛](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/70)（英文关键词直接搜，回复很快） |
| 课程 | NVIDIA DLI 里筛 Jetson / Edge 相关课程（有免费课时） |

**学习顺序建议**：先用 Jetson AI Lab 的 Setup 类教程把环境理顺 → 再做 Fundamentals（Ollama / GenAI）→ 然后 Model Optimization → 最后 Applications（含多模态与 Agent 类项目）。这个目录顺序基本就是本路线。

---

## 5. 现在**不要**碰的东西（避免被劝退）

| 暂时别学 | 什么时候再学 |
|---|---|
| 手写 CUDA C++ kernel | 阶段 3 之后，遇到真实性能瓶颈再学 |
| 设备树 / 驱动移植 / 自研载板 | 除非你要接非官方摄像头或做产品 |
| 从零训练大模型 | 不是边缘 AI 的主战场，先把部署吃透 |
| Isaac Sim 大规模仿真 | 阶段 4 选机器人方向后再碰 |
| 分布式多卡 / 大集群部署 | 与 8GB 单板场景无关 |

---

## 6. 进度追踪表（打印出来打勾）

| 阶段 | 时间 | 交付物 | 完成 |
|---|---|---|---|
| 0 开发环境 | 第 1 周 | 板子体检报告 | ☐ |
| 1 会跑模型 | 第 2~4 周 | 模型性能对比表 | ☐ |
| 2 视觉 + TensorRT | 第 5~10 周 | IMX219 实时检测 demo + 加速比报告 | ☐ |
| 3 大模型 / 多模态 | 第 11~16 周 | 离线多模态问答或语音助手 | ☐ |
| 4 工程化 + 项目 | 第 17~24 周 | GitHub 仓库 + 24h 稳定性验证 | ☐ |

**每周自检三问**
1. 这周我产出了什么能给别人看的东西？
2. 我能不能不看教程重复一遍？
3. 我遇到的报错，我能说出原因吗？

---

## 7. 一句话总结

**先用两周学会"远程开发 + 会跑模型"，再用六周把"摄像头 + TensorRT + 量化"练成肌肉记忆，然后用六周攻下"端侧大模型与多模态"，最后四周把项目工程化到能长期稳定运行。**
全程盯住三件事：**内存、速度、温度**。

---

# 附录 A ｜ SSH + VS Code 远程开发操作流程

> **用途**：配置一次，之后这块板子只需要"电源 + 网线"，全程在你自己的电脑上开发。
> **前置条件**：① 板子和电脑连在**同一个路由器**下（网线或同一个 WiFi）；② 板子现在还能接显示器（因为要开着配置）；③ 知道板子的**用户名和密码**（首次开机时你设置的）。

## A.1 板子端（约 5 分钟，最后一次用显示器和键鼠）

```bash
# 第 1 步：确认 SSH 服务已开启
systemctl status ssh
#   → 显示 active (running) 就跳过下一行
#   → 显示 not found 才需要装：
sudo apt update && sudo apt install -y openssh-server

# 第 2 步：确保开机自动启动
sudo systemctl enable --now ssh

# 第 3 步：查板子的 IP 和用户名，抄到纸上
hostname -I        # 例如 192.168.1.23
whoami             # 例如 nvidia
```

📝 **记录表**（填好贴在显示器边上）

| 项目 | 你的值 |
|---|---|
| 板子 IP | ________________ |
| 用户名 | ________________ |
| 密码 | ________________ |

## A.2 电脑端：第一次连接（Windows PowerShell）

1. 开始菜单搜索并打开 **PowerShell**（Windows 10/11 自带 SSH，不用装软件）
2. 输入（换成你的值）：
   ```powershell
   ssh 用户名@192.168.1.23
   ```
3. 第一次会问 `Are you sure you want to continue connecting?` → 输入 **yes**
4. 输入板子密码（**输入时屏幕上不显示任何字符，这是正常的**，输完按回车）

**成功标志**：命令提示符变成 `用户名@主机名:~$`，说明你已经在板子里面了。

> 提示"无法识别 ssh 命令"：Windows 设置 → 系统 → 可选功能 → 添加功能 → 安装 **OpenSSH 客户端**。

## A.3 验证：现在可以把显示器和键鼠撤掉了

在 SSH 窗口里随便跑两条：

```bash
tegrastats     # 能看到内存/温度/功耗滚动输出（按 Ctrl+C 停止）
ls             # 能看到板子上的文件
```

能跑通，就可以断开显示器，把板子放到任意角落（只要电源和网线接着）。

## A.4 让日常更顺手（三项，可选但强烈推荐）

**① 免密码登录**（在自己电脑的 PowerShell 里执行，只需一次）
```powershell
ssh-keygen -t ed25519                                  # 一路回车
type $env:USERPROFILE\.ssh\id_ed25519.pub | ssh 用户名@192.168.1.23 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
```
之后 `ssh 用户名@192.168.1.23` 直接进入，不再问密码。

**② 防止断线导致长任务被杀**（装 tmux）
```bash
sudo apt install -y tmux
tmux new -s work        # 在里面跑长任务
# 按 Ctrl+B 再按 D 脱离；关掉窗口也没关系
tmux attach -t work     # 下次 ssh 进来，回到刚才的现场
```

**③ 传文件**
- 命令行：`scp 本地文件 用户名@192.168.1.23:/home/用户名/`
- 或者用下面 A.5 的 VS Code，直接拖拽文件树

## A.5 VS Code 远程开发接入（推荐，写完基本不用回终端）

1. 电脑上安装 [VS Code](https://code.visualstudio.com/)
2. 安装扩展 **Remote - SSH**（Microsoft 官方）
3. 按 `F1` → 输入 `Remote-SSH: Connect to Host` → 回车
4. 输入 `用户名@192.168.1.23` → 输入密码
5. 连上后：**文件 → 打开文件夹** → 填 `/home/用户名/`

**效果**：左侧是板子上的文件树，中间写代码，下面终端直接在板子上跑。保存即生效，和本地开发几乎无差别。

## A.6 日常操作流程（每次开工照着走）

1. 给板子通电（SSH 会随系统自动启动，不用按任何键）
2. 电脑打开 VS Code → 连到板子（或直接 PowerShell `ssh`）
3. 先看一眼状态：`tegrastats`（内存够不够、温度高不高）
4. 干活：写代码 / 跑模型 / 装东西
5. 长任务（下载、编译、跑模型）**放进 tmux**，防断线
6. 收工：可以直接断电（板子功耗很低），或 `sudo shutdown -h now` 正常关机

## A.7 连不上时的排查顺序（3 分钟定位）

| 顺序 | 在电脑上做 | 结果判断 |
|---|---|---|
| 1 | `ping 192.168.1.23` | 不通 → 不在同一网络 / IP 变了 / 板子没开机 |
| 2 | 看板子有没有通电、网口灯亮不亮 | 不亮 → 线或电源问题 |
| 3 | 路由器管理页看板子的 IP | 和记录不一致 → **IP 变了**，见下 |
| 4 | 提示 `Connection refused` | 板子上的 ssh 服务没起来（需要接显示器执行 A.1） |

**IP 总变怎么办**：在路由器管理页把板子的 IP 做 **DHCP 保留（静态绑定）**，一劳永逸。
（Jetson 默认跑着 avahi，macOS/Linux 上也可以用 `主机名.local` 访问；Windows 上建议直接用 IP。）

## A.8 常用命令速查（贴在显示器边上）

| 命令 | 作用 | 在哪里执行 |
|---|---|---|
| `ssh 用户名@IP` | 远程登录板子 | 自己电脑 |
| `hostname -I` | 查看板子 IP | 板子 |
| `systemctl status ssh` | 查看 SSH 服务状态 | 板子 |
| `sudo systemctl enable --now ssh` | 开启 SSH 并设为自启 | 板子 |
| `tegrastats` | 看内存 / 温度 / 功耗 / GPU（**最常用**） | 板子 |
| `sudo nvpmodel -q` | 查看当前功耗模式 | 板子 |
| `tmux new -s work` | 开一个防断线会话 | 板子 |
| `scp 文件 用户名@IP:/路径/` | 传文件到板子 | 自己电脑 |
| `sudo reboot` / `sudo shutdown -h now` | 重启 / 关机 | 板子 |
| `exit` | 退出远程登录 | 板子 |

> 注：`nvidia-smi` 在 Jetson 上**不可用**（那是给独立显卡的），请用 `tegrastats`；想要图形化监控可以另外装 `jtop`（jetson-stats 工具包）。

## A.9 首次联网后必做：校准系统时间

Jetson 板子上没有装 RTC 电池，**每次断电重启后时间都会回到 1970-01-01**，联网后由 NTP 自动校准。
所以你在 `systemctl status` 里看到 `since Thu 1970-01-01 ...` 是正常现象，但要确认**当前时间已经正确**：

```bash
date            # 看当前时间对不对
timedatectl     # 看 "System clock synchronized: yes"
```

如果时间还是不对 —— **这会导致 apt / pip / git 报证书错误、Docker 拉镜像失败**：

```bash
sudo timedatectl set-timezone Asia/Shanghai
sudo timedatectl set-ntp true
sudo systemctl restart systemd-timesyncd
date            # 再看一次，确认已是当前时间
```

**养成习惯：每次断电重启后，装东西之前先 `date` 看一眼。**

## A.10 安全提醒

- 只在自己可控的网络内使用，**不要把 22 端口映射到公网**（校园网 / 公司网环境尤其要注意）
- 登录密码不要用出厂默认值
- 公用 WiFi 环境下，别人和你同网段时 SSH 仍是加密的，但别把密码写在共享文档里

---

# 附录 B ｜ 我的板子环境档案（实测，随进展更新）

| 项目 | 实测值 | 查询命令 |
|---|---|---|
| 型号 | Jetson Orin Nano Super 8GB | — |
| 系统 | Ubuntu 22.04.5 LTS | `lsb_release -a` |
| 内核 | 5.15.148-tegra（aarch64） | `uname -a` |
| JetPack / L4T | **L4T R36.4.4 ≈ JetPack 6.2 系**（Ubuntu 22.04 / 内核 5.15） | `cat /etc/nv_tegra_release` |
| 主机名 / 用户名 | nvidia-desktop / **nvidia** | `hostname` / `whoami` |
| **SSH 控制通道** | **USB-C 直连电脑 → 192.168.55.1（固定不变）** | 电脑端 `ssh nvidia@192.168.55.1` |
| 板子上网通道 | 校园网 WiFi → 10.216.255.30 | `hostname -I` |
| 摄像头 | IMX219（CSI） | `ls /dev/video*` |
| 功耗模式 | ✅ 已开启 **MAXN_SUPER**（mode 0，25W 满血） | `sudo nvpmodel -q --verbose` |

> ✅ **USB-C 直连的最大好处：IP 永远是 192.168.55.1，不会像 WiFi 那样变**，省掉了静态绑定这一步。
> ⚠️ 但 USB-C **不提供上网**，板子装包下模型靠它自己的 WiFi。

## 版本对照表（决定跟哪套教程）

| JetPack | L4T | Ubuntu | 内核 | 说明 |
|---|---|---|---|---|
| 5.1.x | 35.x | 20.04 | 5.10 | 老旧教程 |
| **6.0** | 36.3 | 22.04 | 5.15 | |
| **6.1** | 36.4 | 22.04 | 5.15 | |
| **6.2+** | 36.4.3+ | 22.04 | 5.15 | **开始支持 Orin Nano Super 25W 模式** |
| 7.x | 39.x | 24.04 | 6.8 | 最新；已取消 SD 卡镜像，教程较少 |

**结论：本机是 6.x 系，找教程时优先选「JetPack 6.x / L4T r36.x」，不要照抄 7.x 的（刷机、容器、库版本都不一样）。**

## 首次登录后要做的四件事

```bash
cat /etc/nv_tegra_release              # 1. 记录确切版本
date                                   # 2. 校时（不对则见 A.9）
sudo nvpmodel -q --verbose             # 3. 确认功耗模式里有 MAXN SUPER
passwd                                 # 4. 改掉卖家给的密码（可选但建议）
```

## 系统更新（894 个更新待装）

```bash
sudo apt update
sudo apt upgrade -y            # 耗时较长，建议放进 tmux
sudo reboot                    # 更新后需要重启
```
> 注意：更新前确认散热正常、电源稳定；更新期间不要断电。

## 本机已验证的状态（2026-10-15 实测）

- ✅ **系统时间正确**（2026-10-15 13:07 CST，NTP 已同步）。`systemctl status` 里出现 1970 只是"开机瞬间还没联网校时"，无需处理。
- ✅ **功耗模式已是 MAXN_SUPER**（mode 0）——Super 满血模式已开启，**不需要改任何设置**。
- ✅ **`nvpmodel --verbose` 里的 NVPM ERROR 是正常现象**：报错集中在 `cpu6/cpu7` 和 `PVA0` —— Orin Nano 只有 **6 个 CPU 核心**、**没有 PVA/DLA**（那是 Orin NX/AGX 才有的），配置文件按高端型号写所以会报"打不开"，不影响功能。DLA 那几行会优雅跳过（SKIP）。
- 📌 参考值（Super 模式）：GPU 最大频率 1.02 GHz，EMC 3199 MHz（内存带宽约 102 GB/s）。

## 待排查：passwd 报「认证令牌操作错误」

含义：系统无法写入 `/etc/shadow`。常见三种原因（**第 1 条最要紧，它会让后面的 apt 安装全部失败**）：

```bash
mount | grep ' / '      # ① 根文件系统是不是 ro（只读）
df -h /                 # ② 磁盘是不是满了
df -i /                 # ③ inode 是不是耗尽
ls -l /etc/shadow       # 应为 -rw-r----- 1 root shadow
dmesg | tail -30        # 看有没有 ext4 / I/O 报错
```

若发现根文件系统是 `ro`：

```bash
sudo mount -o remount,rw /
sudo passwd nvidia      # 再试一次
```



