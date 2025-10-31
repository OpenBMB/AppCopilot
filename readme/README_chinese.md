# AppCopilot:多模态多智能体驱动的通用型跨应用端侧助理

## 🌐 语言选择

<p align="center">
    【English| <a href="readme/README-Chinese.md">Chinese</a>】
</p>
## 🎨 Logo

![Emulator Demo](images/logo.png)

## 📖 简介

AppCopilot是一款以多模态大模型为基座、融合多智能体协作机制的通用型端侧智能助理。借助强大的多模态理解与生成能力，AppCopilot可理解文本、图像等多源信息，并作为智能中枢编排多个智能体协同完成复杂任务。通过端侧部署，AppCopilot不仅保障了用户数据隐私与本地实时性，还能在不同应用之间无缝切换，实现跨App操作和智能联动。更进一步，AppCopilot支持跨设备协作，让多手机甚至与PC等多终端之间形成智能互联，实现任务流转与信息同步。该作品展示了基于多模态大模型和多智能体技术，如何构建灵活、高效、安全的下一代通用型数字助理，赋能各类智能终端创新体验。

## ⚡️ 复现指南

<details>
<summary>Click to expand</summary>

### AppCopilot 本地运行

本节主要介绍如何通过 API 连接服务器上已经训好的模型，在本地运行 AppCopilot。

#### 本地环境基本要求

表格展示了本地环境的相关依赖要求：

| **依赖项**    | **具体要求**                                                   |
|---------------|--------------------------------------------------------------|
| 操作系统      | 支持 Android Studio 运行的操作系统                               |
| 软件          | 安装 Android Studio                                            |
| Python 环境   | 安装 Python 环境，建议安装版本号为 3.12 的 Python 版本        |
| 网络          | 关闭本地 VPN，确保服务器端 vllm api 的正常连接                |

##### 安装 Android Studio

Android Studio 是一个为 Android 平台开发程序的集成开发环境。可通过其官网 [Android Studio 官网](https://developer.android.com/studio) 下载。

#### 服务器环境基本要求

表格介绍了服务器端环境的相关依赖要求：

| **依赖项**    | **具体要求**                                                   |
|---------------|--------------------------------------------------------------|
| 操作系统      | 支持 Conda 和 vLLM 运行的操作系统                             |
| 软件          | 安装 Conda 并创建 vLLM 环境、安装 vLLM 相关依赖              |

##### Conda 安装

Conda 是一个开源的跨平台包管理器和环境管理器，它能够帮助用户快速安装、运行和管理包含多种语言的软件包及其依赖项。可以通过其官网 [Conda 官网](https://anaconda.org/anaconda/conda) 下载。

安装好 Conda 后，配置 Python 虚拟环境，推荐 Python 版本号为 3.12。

```bash
conda create --name vllm_env python=3.12
```

##### vLLM 安装

[vLLM](https://docs.vllm.ai/en/latest/) 是一个用于大语言模型推理和服务的开源高性能库，以更低的成本和更高的效率，为生成式AI应用提供更快的响应。 此处需要配置 vLLM 相关环境依赖，使用如下命令安装版本为 0.9.1 的 vLLM：

```bash
pip install vllm==0.9.1
```
##### 其余配置
要通过 API 连接服务器运行 AppCopilot，服务器环境其他配置要求如下：

```bash
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils
pip install openai
git clone https://huggingface.co/Qwen/Qwen-VL-7B
```

#### 代码克隆
首先，从远程克隆文件夹到本地，并添加相关文件：

```bash
mkdir AppCopilot
cd AppCopilot
git clone https://github.com/OpenBMB/AppCopilot.git .
```

为了提升智能体在安卓手机上的操作能力，本项目还需安装 YADB 工具以增强原生 ADB 功能。它解决了 ADB 在文本输入、截屏和 UI 布局提取等方面的局限性，提供了更高效、更精确的操作。执行以下命令：

```bash
git clone https://github.com/ysbing/YADB.git ./YADB
```

#### 本地配置系统环境变量
##### 配置 adb 环境变量
1.Windows 系统 adb 环境变量配置:

在 Windows 系统中，右键点击【此电脑】，选择【属性】，点击【高级系统设置】。

在弹出的窗口中点击【环境变量】，在系统变量中点击【新建】，输入变量名：adb，变量值添加 adb 所在的目录路径（例如：`C:\Android\Sdk\platform-tools`），接着在系统变量中找到 Path，向其中添加刚刚添加的 adb 环境。双击 Path，点击【新建】输入 %adb%。

2.macOS/Linux 系统 adb 环境变量配置

在 Linux 或 macOS 系统中，编辑 ~/.bashrc 或 ~/.bash_profile 文件，在文件末尾添加 adb 路径：
```bash
/Users/user/Android/Sdk/platform-tools
```
保存文件后，执行 source ~/.bashrc 或 source ~/.bash_profile 使配置生效。

完成以上配置后，在命令行输入 `adb version`，若能正确输出 adb 版本号等相关信息，即代表配置成功。

##### 配置 emulator 环境变量
配置方法与配置 adb 环境变量的方法类似。

1.Windows 系统 emulator 环境变量配置

在 Windows 系统中，右键点击【此电脑】，选择【属性】，点击【高级系统设置】。

在弹出的窗口中点击【环境变量】，在系统变量中点击【新建】，输入变量名：emulator，变量值添加 emulator 所在的目录路径（例如：`C:\Android\Sdk\emulator`），接着在系统变量中找到 Path，向其中添加刚刚添加的 emulator 环境。双击 Path，点击【新建】输入 %emulator%。

2.macOS/Linux 系统 emulator 环境变量配置

在 Linux 或 macOS 系统中，编辑 ~/.bashrc 或 ~/.bash_profile 文件，在文件末尾添加 emulator 路径：
```bash
/Users/user/Library/Android/Sdk/emulator
```
保存文件后，执行source ~/.bashrc或source ~/.bash_profile使配置生效。
完成以上配置后，在命令行输入`emulator version`，若能正确输出emulator版本号等相关信息，即代表配置成功。

#### 配置用于运行的安卓设备
##### 配置 emulator 环境变量
本项目使用 Android Studio 创建和管理安卓虚拟设备（Android Virtual Device，AVD）。可以参考 Android Studio 官方文档配置虚拟器。

查看模拟器名称和列表：在命令行中输入命令 `emulator -list-avds` 来查看目前的模拟器名称和列表，后续可指定开启某个模拟器。

配置模拟器网络：在命令行中输入命令：
```bash
emulator -avd <android> -dns-server <Local DNS Server>
```
其中 `<android>` 是指定的模拟器名称，`<Local DNS Server>` 是本地 DNS 地址。仅第一次需要指定 DNS Server，之后可直接启动：`emulator -avd <android>`,。如
果在调试过程中出现了快照损坏的报错，可以在启动时加上-no-snapshot-load的参数后缀。

在完成上述配置后，安卓模拟器应可在本地正常运行，呈现可交互的图形界面，支持鼠标操作，同时通过主机网络共享实现网络访问。

##### 配置实体机
除了使用安卓虚拟机（AVD）之外，智能体还可以通过 adb 操作实体手机。下面列出使用 adb 操作实体手机的具体步骤：

打开安卓实体机开发者模式：进入手机【设置】->【我的设备】->【全部参数和信息】->点击【MIUI 版本】7 次，进入手机开发者模式。

启用 USB 调试模式：在手机【设置】中找到【开发者选项】，启用【USB 调试】。

使用 adb 连接实体机：通过数据线将电脑和实体机连接，在命令行输入命令 adb devices，若能看到实体机对应序列号，代表连接成功。

##### 配置 Python 相关环境依赖
推荐安装并使用版本号为 3.12 的 Python 版本。本地进入之前克隆的 GUI-Android 目录，安装如下的依赖项：
```bash
pip install -r requirements.txt
```

##### 配置相关模型密钥
在本地代码文件 `./wrappers/constants.py` 中，需要用户手动配置 LLM 密钥，以便后续模型调用过程。
```bash
# ----- model config -----
MODEL_EXTRACT = "AppCopilot"
ERROR_CALLING_LLM = "Error calling LLM"
MODEL_NOT_FOUND = "LLM not found"

# 此处需改为本地实际监听端口
END_POINT = "http://localhost:8001/v1/chat/completions"
PORTS = [8002, 8003, 8004]

# 此处需要换成用户提供的 API 密钥和 Base URL
CLIENT_API_KEY = "switch to your own api key"
CLIENT_BASE_URL = "switch to your own base url"
CLIENT = OpenAI(api_key=CLIENT_API_KEY, base_url=CLIENT_BASE_URL)
```

##### 下载AppCopilot模型
从`https://huggingface.co/ffcosmos/AppCopilot/tree/main`下载已经训练好的AppCopilot模型，放在服务器，以便接下来启动vLLM推理服务。

##### 服务器端 vLLM 服务启动
为实现 AppCopilot 对本地大语言模型的远程调用能力，需在服务器端预先部署并启动 vLLM 推理服务。

服务器端GUI模型vLLM服务启动:
```bash
#/your/model/path替换为实际的GUI模型路径
vllm serve /your/model/path \
  --served-model-name AppCopilot \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt image=10 \
  --max_model_len 2048 \
  --port 8001
```

服务器端Qwen2.5-VL-7B-Instruct模型vLLM服务启动:
```bash
#/your/model/path替换为实际的Qwen2.5-VL-7B-Instruct模型路径
vllm serve /your/model/path \
  --served-model-name Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 1 \      
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --port 8002 
```

#### 本地运行启动AppCopilot
在本地启动程序之前，应首先将远程服务器上的8001端口转发至本地的8001端口，
将远程服务器上的8002端口转发至本地的8002端口，以确保本地环境能够通过HTTP接口访问服务器端的模型服务。该端口转发操作可通过本地终端执行相应命令实现。
```bash
ssh -L 8001:localhost:8001 username@model-server-ip
ssh -L 8002:localhost:8002 username@model-server-ip 
```

##### 单端运行
最后是最终的 AppCopilot 运行。若要在单设备单端运行，在本地终端中打开命令行界面，进入包含 `run_agent.py` 文件的目录。随后，依据命令行参数说明表（见下文），传入所需功能对应的参数运行 `run_agent.py` 脚本，即可完成 AppCopilot 的启动。

以下是示例命令，启用语音输入以及语音反馈，并运行自定义任务：

```bash
# 启用语音输入以及语音反馈，并运行自定义任务
python run_agent.py --custom-task 
```

| 参数                          | 类型   | 说明                                        |
| ----------------------------- | ------ | ------------------------------------------- |
| `--predefined-task <TASK_NAME>` | str    | 指定预定义任务的名称（任务名需在内置列表中）。 |
| `--custom-task`                | flag   | 启用自定义任务模式，跳过预定义任务选择。   |
| `--enable-experience`          | flag   | 启用基于经验的任务匹配机制。               |
| `--enable-voice-input`         | flag   | 启用语音输入（仅在自定义任务模式下有效）。 |
| `--enable-audio`               | flag   | 启用音频反馈。                             |
| `--show-tasks`                 | flag   | 显示所有可用的预定义任务并退出程序。       |
| `--enable-vision-parser`       | flag   | 是否调用 omniparser 进行坐标校准。         |
| `--read-final-page`            | flag   | 是否启用朗读最终界面。                     |

##### 多设备跨端运行

如果需要进行多设备跨端场景的运行，进入包含 `cross_device_agent.py` 的目录，随后，依据命令行参数说明表，传入所需功能对应的参数运行脚本，即可完成 AppCopilot 多设备跨端的启动。

| 参数                  | 类型  | 说明                                         |
| --------------------- | ----- | -------------------------------------------- |
| `--device1-serial`     | str   | 设备1的ADB序列号（可选）                    |
| `--device1-port`       | int   | 设备1的通信端口（默认11001）。              |
| `--device2-serial`     | str   | 设备2的ADB序列号（可选）                    |
| `--device2-port`       | int   | 设备2的通信端口（默认11002）。              |
| `--task`               | str   | 跨设备任务指令。                            |

### 模型推理评测
#### 数据准备
##### Android Control

下载[Android Control](https://github.com/google-research/google-research/tree/master/android_control)并保存在 ``/eval/eval_data/tmp/android_control``

```
cd eval/eval_data
python process_ac.py
ln -s android_control_test android_control_high_test
ln -s android_control_test android_control_low_test
```

##### CAGUI

```
cd eval/eval_data
mkdir chinese_app_test && cd chinese_app_test
huggingface-cli download openbmb/CAGUI --repo-type dataset --include "CAGUI_agent/**" --local-dir ./ --local-dir-use-symlinks False --resume-download
mv CAGUI_agent test
```

##### aitz

下载 [aitz](https://github.com/IMNearth/CoAT)并保存在 ``/eval/eval_data/tmp/android_in_the_zoo``

```
cd eval/eval_data
mv tmp/android_in_the_zoo ./aitz_test
python process_aitz.py
```

##### gui-odyssey

下载[GUI-Odyssey](https://github.com/OpenGVLab/GUI-Odyssey?tab=readme-ov-file)并保存在 ``/eval/eval_data/tmp/GUI-Odyssey``. 从GUI-Odyssey仓库复制 [preprocessing.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/preprocessing.py) 和 [format_converter.py](https://github.com/OpenGVLab/GUI-Odyssey/blob/master/data/format_converter.py) 到 ``/eval/eval_data/tmp/GUI-Odyssey``

```
cd eval/eval_data/tmp/GUI-Odyssey
python preprocessing.py
python format_converter.py
python ../../process_odyssey.py
```

#### 进行推理
模型评测阶段所需要运行的程序都集成在`eval_multi.sh`脚本中。在执行前，请根据实际数据存储位置，修改脚本中涉及的路径参数，以确保文件的正确加载与处理。
```bash
# 在eval.sh中需要修改的内容
# Configure basic parameters
data_name="evaluation dataset"
model_name="target model name"
base_output_dir="result directory"

# List of models to process
models_base_path=(
    "models base path"
)
```
在执行模型推理评测前，请确保服务器环境中`utils`文件夹已正确配置。完成路径参数的正确设置后，可在命令行中执行相应程序以启动模型推理评测流程。

命令行执行评测命令：
```bash
bash eval_multi.sh
```

</details>

## ✨ **展示案例**

<details>
<summary>点击展开</summary>


### Case 1: 长程任务

![Long Horizon Demo](C:\Users\Administrator.DESKTOP-RN0CUUV\Desktop\readme\images\long_horizon.png)

### Case 2: 跨端任务

![Cross Device Demo](C:\Users\Administrator.DESKTOP-RN0CUUV\Desktop\readme\images\double_end.png)

### Case 3: 三端任务

![Triple end Demo](C:\Users\Administrator.DESKTOP-RN0CUUV\Desktop\readme\images\triple_end.png)

</details>

## **⚖️ 授权许可**

- **源代码授权**：本项目的源代码采用 Apache 2.0 许可证。该许可证允许在遵守 Apache 2.0 条款的前提下使用、修改及分发代码。
- **数据授权**：项目相关数据遵循 CC BY-NC 4.0 许可协议。此协议明确规定数据仅限非商业用途。需特别注意的是，基于这些数据集训练的模型必须严格遵循非商用限制，且仅可用于研究目的。

## **📬 联系我们**

如有任何疑问、建议或合作意向，欢迎通过邮件联系：[qianc62@gmail.com](mailto:qianc62@gmail.com)
