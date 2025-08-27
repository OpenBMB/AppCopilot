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

### AppCopilot本地运行
本节主要介绍如何通过API连接服务器上已经训好的模型，在本地运行 AppCopilot。

#### 本地环境基本要求
| 依赖项 | 具体要求 |
| --- | --- |
| 操作系统 | 支持 Android Studio 运行的操作系统 |
| 软件 | 安装 Android Studio |
| Python 环境 | 安装 Python 环境，建议安装版本号为 3.12 的 Python 版本 |
| 网络 | 关闭本地 VPN，确保服务器端 vllm api 的正常连接 |

###### 安装 Android Studio
Android Studio是一个为 Android 平台开发程序的集成开发环境。可通过其官网 <https://developer.android.com/studio> 下载。

#### 服务器环境基本要求
| 依赖项 | 具体要求 |
| --- | --- |
| 操作系统 | 支持 Conda 和 vLLM 运行的操作系统 |
| 软件 | 安装 Conda 并创建 vLLM 环境、安装 vLLM 相关依赖 |

###### Conda 安装
Conda是一个开源的跨平台包管理器和环境管理器，它能够帮助用户快速安装、运行和管理包含多种语言的软件包及其依赖项。可以通过其官网<https://anaconda.org/anaconda/conda> 下载。

安装好 Conda 后，配置 Python 虚拟环境，推荐 Python 版本号为 3.10。

```bash
conda create --name vllm_env python=3.10
```

###### vLLM 安装
vLLM(<https://docs.vllm.ai/en/latest/>)是一个用于大语言模型推理和服务的开源高性能库，以更低的成本和更高的效率，为生成式AI应用提供更快的响应。 此处需要配置 vLLM 相关环境依赖，使用如下命令安装版本为 0.9.1 的vLLM：
```bash
pip install vllm==0.9.1
```

###### 其余配置

想要本地通过API连接服务器运行AppCopilot，服务器环境其余配置要求如下代码块

```bash
pip install git+https://github.com/huggingface/transformers@f3f6c86582611976e72be054675e2bf0abb5f775
pip install accelerate
pip install qwen-vl-utils
pip install openai
git clone https://huggingface.co/Qwen/Qwen-VL-7B
```

#### 代码克隆

首先，从远程克隆文件夹到本地，并添加相关文件。

```bash
mkdir AppCopilot
cd AppCopilot
git clone https://github.com/GUIAgents-Dev/GUI-Android.git .
```
为了提升智能体在安卓手机上的操作能力，本项目还需安装 YADB 工具以增强原生 ADB（Android Debug Bridge）功能。它解决了 ADB 在文本输入、截屏和 UI 布局提取等方面的局限性，提供了更高效、更精确的操作。在当前目录下执行命令：

```bash
git clone https://github.com/ysbing/YADB.git ./YADB 
```

#### 本地配置系统环境变量
##### 配置 adb 环境变量
- **Windows 系统 adb 环境变量配置**：在 Windows 系统中，右键点击【此电脑】，选择【属性】，点击【高级系统设置】，在弹出的窗口中点击【环境变量】，在系统变量中点击【新建】，输入变量名：adb，变量值添加 adb 所在的目录路径（例如：C:\Android\Sdk\platform-tools），接着在系统变量中找到 Path，向其中添加刚刚添加的 adb 环境。双击 Path ，点击【新建】输入 【%adb%】。
- **macOS/Linux 系统 adb 环境变量配置**：在 Linux 或 macOS 系统中，编辑 `source ~/.bashrc` 或 `source ~/.bash_profile`文件, 在文件末尾添加 adb 路径： `/Users/user/Android/Sdk/platform-tools`，保存文件后，执行 `source ~/.bashrc`或 `source ~/.bash_profile`使配置生效。

完成以上配置后，在命令行输入 `adb version`，若能正确输出 adb 版本号等相关信息，即代表配置成功。

##### 配置 emulator 环境变量

配置方法与上面配置 adb 的环境变量的方法较为类似。

- **Windows 系统 emulator 环境变量配置**：在 Windows 系统中，右键点击【此电脑】，选择【属性】，点击【高级系统设置】，在弹出的窗口中点击【环境变量】，在系统变量中点击【新建】，输入变量名：emulator，变量值添加 emulator 所在的目录路径（例如：C:\Android\Sdk\emulator），接着在系统变量中找到 Path，向其中添加刚刚添加的 emulator 环境。双击 Path ，点击【新建】输入 【%emulator%】。
- **macOS/Linux 系统 adb 环境变量配置**：在 Linux 或 macOS 系统中，编辑 `source ~/.bashrc` 或 `source ~/.bash_profile`文件, 在文件末尾添加 emolator 路径： `/Users/user/Library/Android/Sdk/emulator`，保存文件后，执行 `source ~/.bashrc`或 `source ~/.bash_profile`使配置生效。

完成以上配置后，在命令行输入 `adb version`，若能正确输出 adb 版本号等相关信息，即代表配置成功。

#### 配置用于运行的安卓设备
##### 配置模拟器

本项目使用 Android Studio 创建和管理安卓虚拟设备（Android Virtual Device，AVD），可参考 Android Studio 的官方文档配置虚拟器https://developer.android.com/studio/run/managing-avds

1. **查看模拟器名称和列表**：在命令行中输入命令：

```bash
emulator -list-avds
```

来查看目前的模拟器名称和列表，后续可指定开启某个模拟器。

   2.**配置模拟器网络**：在命令行中输入命令：

```
emulator -avd <android> -dns-server  <Local DNS Server>
```

其中 <android> 是指定的模拟器名称，<Local DNS Server> 是本地 DNS 地址。仅第一次需要指定 DNS Server，之后可直接启动：emulator -avd <android>。如果在调试过程中出现了快照损坏的报错，可以在启动时加上 -no-snapshot-load的参数后缀。

  在完成上述配置后，安卓模拟器应可在本地正常运行，呈现可交互的图形界面，支持鼠标操作，同时通过主机网络共享实现网络访问。 下图展示了在启动安卓虚拟机后的项目页面截图

![Emulator Demo](images/emunew.png)

##### 配置实体机

除了使用安卓虚拟机（AVD）之外，智能体还可以通过 adb 操作实体手机。下面列出使用 adb 操作实体手机的具体步骤

1. **打开安卓实体机开发者模式**：以 小米手机 MIUI 14.0.11 版本为例，进入手机【设置】，点击【我的设备】，下滑点击【全部参数信息】，点击【MIUI 版本】7 次，进入手机开发者模式。
2. **启用 USB 调试**：在手机【设置】中找到【开发者选项】，下滑找到【USB 调试】并启用该功能，启用后实体机可通过连接 USB 后启用调试模式，让 adb 能够进行模拟操作。
3.  **使用 adb 连接实体机**：在完成上一步操作后，使用数据线将电脑和实体机相连接，在电脑命令行输入命令 adb devices, 如果出现实体机对应序列号，例如3e90f1ef device，即说本地端与实体机端已经通过 adb 建立连接，配置完成。



#### 配置 Python 相关环境依赖

推荐安装并使用版本号为 3.12 的 Python 版本。本地进入之前克隆的 GUI-Android目录, 安装如下的依赖项：

```bash
pip install -r requirements.txt 
```

#### 配置相关模型密钥

在本地代码文件 ./wrappers/constants.py 中，需要用户手动配置 LLM 密钥，以便后续模型调用过程。代码块 8 展示了更改具体配置的位置和变量名。

```python
# ----- model config -----
MODEL_EXTRACT = "deepseek-v3-250324"
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

#### 服务器端vLLM服务启动

为实现 AppCopilot 对本地大语言模型的远程调用能力，需在服务器端预先部署并启动 vLLM 推理服务。该服务通过 HTTP API 提供模型访问接口，需在命令行中执行启动命令，并根据实际情况将模型路径参数设置为已训练模型的存储目录。我们需要把服务器中已训练好的 GUI 模型和下载的 Qwen-VL-7B 启动 vLLM 服务, 分别部署到 8001和 8002 端口。

```bash
#/your/model/path替换为实际的GUI模型路径
vllm serve /your/model/path \
  --served-model-name AgentCPM-GUI \
  --tensor_parallel_size 1 \
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --limit-mm-per-prompt image=10 \
  --max_model_len 2048 \
  --port 8001 
```
```bash
#/your/model/path替换为实际的Qwen-VL-7B模型路径
vllm serve /your/model/path \
  --served-model-name AgentCPM-GUI \
  --tensor-parallel-size 1 \      
  --trust-remote-code \
  --gpu-memory-utilization 0.9 \
  --port 8002 
```

#### 本地运行启动AppCopilot

在本地启动程序之前，应首先将远程服务器上的 8001 端口转发至本地的 8001 端口，将远程服务器上的 8002 端口转发至本地的 8002 端口，以确保本地环境能够通过 HTTP接口访问服务器端的模型服务。该端口转发操作可通过本地终端执行相应命令实现

```bash
ssh -L 8001:localhost:8001 username@model-server-ip
ssh -L 8002:localhost:8002 username@model-server-ip 
```

##### 单端运行

最后是最终的 AppCopilot 运行。若要在单设备单端运行，在本地终端中打开命令行界面，进入包含 run_agent.py 文件的目录。随后，依据命令行参数说明表（见表 8.3），传入所需功能对应的参数运行 run_agent.py 脚本，即可完成 AppCopilot 的启动。以下代码块12为示例命令，启用语音输入以及语音反馈，并运行自定义任务：

```bash
# 启用语音输入以及语音反馈，并运行自定义任务
python run_agent.py --custom-task --enable-voice-input --enable-audio 
```

**命令行参数说明**

| 参数                            | 类型 | 说明                                           |
| ------------------------------- | ---- | ---------------------------------------------- |
| `--predefined-task <TASK_NAME>` | str  | 指定预定义任务的名称（任务名需在内置列表中）。 |
| `--custom-task`                 | flag | 启用自定义任务模式，跳过预定义任务选择。       |
| `--enable-experience`           | flag | 启用基于经验的任务匹配机制。                   |
| `--enable-voice-input`          | flag | 启用语音输入（仅在自定义任务模式下有效）。     |
| `--enable-audio`                | flag | 启用音频反馈。                                 |
| `--show-tasks`                  | flag | 显示所有可用的预定义任务并退出程序。           |
| `--enable-vision-parser`        | flag | 是否调用 omniparser 进行坐标校准。             |
| `--read-final-page`             | flag | 是否启用朗读最终界面。                         |

##### 多设备跨端运行

如果需要进行多设备跨端场景的运行，进入包含 cross\_device\_agent.py 的目录，随后，依据命令行参数说明表，传入所需功能对应的参数运行 cross\_device\_agent.py 脚本，即可完成 AppCopilot 多设备跨端的启动。

**命令行参数说明**

| 参数               | 类型 | 说明                              |
| ------------------ | ---- | --------------------------------- |
| `--device1-serial` | str  | 设备 1 的 ADB 序列号（可选）      |
| `--device1-port`   | int  | 设备 1 的通信端口（默认 11001）。 |
| `--device2-serial` | str  | 设备 2 的 ADB 序列号（可选        |
| `--device2-port`   | int  | 设备 2 的通信端口（默认 11002）。 |
| `--task`           | str  | 跨设备任务指令。                  |

### 服务器上进行模型后训练

本节主要介绍如何复现在服务器上进行模型后训练的完整流程，包括数据预处理、监督微调（Supervised Fine-Tuning, SFT）、强化微调（Reinforcement Fine-Tuning,RFT），以及在后训练完成后的模型推理评测。

#### 数据预处理

项目在对模型进行后训练之前，需要先对收集到的 GUI 交互数据进行预处理。整个数据处理流程主要包括三部分：首先对原始数据进行清洗，移除不符合质量标准的样本；其次将有效数据转换为统一的结构化训练格式；最后通过数据增强方法扩充数据规模以提高模型的泛化能力。

##### 数据清洗

数据清洗过程通过已提供的 clear.py 脚本完成。该脚本所依赖的均为 Python 标准库模块，因而在已正确安装 Python 的前提下，无需额外配置运行环境。在执行前，请根据实际数据存储位置，修改脚本中主程序入口处涉及的路径参数 ，以确保文件的正确加载与处理。

```python
if __name__ == "__main__":
    main_folder = "/your/path1" #替换为待清洗的数据的路径 
    tmp_folder = "/your/path2" #用于存放那些 instruction 字段重复的数据
    tmp_step_folder = "/your/path3" #用于存放 path 长度不符合要求的数据
```
路径参数修改完毕后，在命令行运行程序即可进行数据清洗:

```bash
python clear.py 
```

##### 数据格式标准化

数据清洗过程通过已提供的 data.py 脚本完成。同样的，在执行前，请根据实际数据存储位置，修改脚本中 main 函数涉及的路径参数，以确保文件的正确加载与处理

```python
def main():
    """主函数，输出处理后的文件夹数量"""
    source_base = "/your/path1" #替换为待进行格式转换的数据目录
    destination_base = "/your/path2" #转换后数据的输出目录
```
路径参数修改完毕后，在命令行运行程序即可进行数据格式标准化:

```bash
python data.py 
```

##### 数据增广

数据清洗过程通过已提供的 data_process_ins.py 脚本完成。同样的，在执行前，请根据实际数据存储位置，修改脚本中涉及的路径参数，以确保文件的正确加载与处理。

```python
#调用的模型替换成实际使用的模型
client = OpenAI(
  api_key='your_api_key',
  base_url='your_base_url'
) 
model_name = "your_model"

source_base = "/your/path1" #替换为待增广的数据目录
destination_base = "/your/path2" #增广后数据的输出目录
```
路径参数修改完毕后，在命令行运行程序即可进行数据增广:

```bash
python data_process_ins.py 
```

#### 监督微调
##### 环境配置

SFT 阶段所需要运行的程序都集成在 finetune_ds.sh 脚本中。首先先进入 SFT 相关目录中，配置用于 SFT 的环境。

```bash
# conda 新建环境
conda create -n gui-sft python=3.10
# 激活 conda 环境
conda activate gui-sft
# pip安装包
# 注意：此处要把requirements.txt 中的 flash-attn 先注释掉再安装
pip install -r requirements.txt 
# 单独安装 flash-attn，必须要指定版本为 2.7.4.post1
pip install flash_attn==2.7.4.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
```

##### 运行SFT脚本

同样的，在执行前，请根据实际数据存储位置，修改脚本中涉及的路径参数，以确保文件的正确加载与处理。

```bash
MODEL="/path/to/your/model" #替换为你的预训练模型
# or openbmb/MiniCPM-V-2, openbmb/MiniCPM-Llama3-V-2_5, openbmb/MiniCPM-V-2_6

# ATTENTION: specify the path to your training data, which should be a json file consisting of a list of conversations.
# See the section for finetuning in README for more information.
DATA="/your/path/to/data"
EVAL_DATA="/your/path/to/eval_data" #替换为你的数据路径

# if use openbmb/MiniCPM-V-2, please set LLM_TYPE=minicpm, if use openbmb/MiniCPM-Llama3-V-2_5, please set LLM_TYPE="llama3",
# if use openbmb/MiniCPM-o-2_6 or openbmb/MiniCPM-V-2_6, please set LLM_TYPE=qwen
LLM_TYPE="qwen" #根据实际模型替换LLM_TYPE
```

路径参数修改完毕后，在命令行运行程序即可进行 SFT:

```
bash finetune_ds.sh
```



#### 强化微调

##### 环境配置

RFT 阶段所需要运行的程序都集成在 fsdp.sh 脚本中。首先先进入 RFT 相关目录中，配置用于 RFT 的环境。

```bash
#conda新建环境
conda create -n fjr-arl python=3.11
#激活环境
conda activate fjr-arl
#进入./AgentCPM-GUI路径pip安装包，把flash_attn/torch/transformers注释掉
pip install -r requirements.txt
#进入./AgentCPM-GUI/rft路径pip安装包 把flash_attn/torch/transformers注释掉
pip install -r requirements.txt
#pip 单独安装一些指定版本的包
#单独安装 flash-attn，指定版本为 2.7.4.post1
pip install flash_attn==2.7.4.post1 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-build-isolation
#根据cuda版本下载对应的torch，注意torch版本需>=2.6.0，例如cuda12.4对应的torch下载命令如下：
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
#单独安装transformers，指定版本为4.51.0
pip install transformers==4.51.0
note：检查torch版本>=2.6.0以及transformers的版本为4.51.0才能run起来FSDPv2。
```

##### 运行RFT脚本

同样的，在执行前，请根据实际数据存储位置，修改脚本中涉及的路径参数，以确保文件的正确加载与处理。

```bash
source /opt/miniconda3/bin/activate fjr-arl #替换为实际miniconda环境路径
...
--model_name_or_path /path/to/your/model #替换为实际模型路径
--dataset_name /path/to/your/dataset #替换为实际训练数据路径
--eval_dataset_name /path/to/your/eval_dataset #替换为实际测试数据路径
```

在执行强化微调（RFT）流程前，请确保服务器环境中 trainer/utils 文件夹以及trainer/arl.py 已正确配置，并与 fsdp.sh 与 grpo.py 文件处于同一目录层级。若路径配置不当，程序将无法正常加载所依赖的模块，导致运行失败。完成路径参数的正确设置后，可在命令行中执行相应程序以启动 RFT 流程。

```
 bash fsdp.sh
```



#### 模型推理评测

模型评测阶段所需要运行的程序都集成在 eval.sh 脚本中。同样的，在执行前，请根据实际数据存储位置，修改脚本中涉及的路径参数，以确保文件的正确加载与处理。

```bash
model_base_path="/path/to/your/model" #替换为待评测模型路径
data_name="/path/to/your/data" #评测数据集
model_name="your_model_name" #替换为待评测模型
base_output_dir="/path/to/your/output" #结果输出路径
```

在执行模型推理评测前，请确保服务器环境中 utils 文件夹已正确配置，并与 eval.sh与 run_predict_minicpm.py，run_eval_agent.py 文件处于同一目录层级。若路径配置不当，程序将无法正常加载所依赖的模块，导致运行失败。完成路径参数的正确设置后，可在命令行中执行相应程序以启动模型推理评测流程。

```
bash eval.sh 
```



### 资源汇总

项目在数据处理、模型后训练以及评测阶段均配备了结构清晰、功能明确的支撑性脚本与配置文件，确保整个系统在构建、训练与评估流程中具备良好的可控性与可重复性。在数据处理部分，相关脚本用于实现数据清洗、格式转换及数据增强，支撑多模态训练数据的标准化构建；在后训练阶段，涵盖了监督微调（SFT）与强化微调（RFT）两个关键流程，所对应的训练脚本与配置文件能够有效支持模型的多阶段优化与能力提升；而在评估阶段，则集成了自动化推理与指标计算模块，实现了模型性能的系统化、标准化评估。上述各阶段所使用的核心文件与数据统计详见表 8.5、表 8.6、表 8.7 及表 8.8，为系统构建与实验复现提供了重要保障。

#### 数据处理相关文件

数据处理流程主要包括三部分：首先对原始数据进行清洗，移除不符合质量标准的样本；其次将有效数据转换为统一的结构化训练格式；最后通过数据增强方法扩充数据规模以提高模型的泛化能力。所使用的具体文件详见表 8.5。各训练阶段所使用的训练数据量详见下表。

| 文件名 | 格式 | 描述 |
| --- | --- | --- |
| clear.py | Python | 清洗原始数据 |
| data.py | Python | 转化为结构化数据 |
| data_process_ins.py | Python | 指令文本增强 |
| data_process_bbox.py | Python | 边界框数据增广 |

**Dataset Sizes**:

| 训练阶段                          | 数据量 |
| --------------------------------- | ------ |
| 继续预训练增强 GUI Grounding 能力 | 1200万 |
| 监督微调 SFT                      | 600万  |

#### 后训练相关文件

后训练过程包括两个阶段：首先进行监督微调，随后进行强化微调。在 SFT 阶段，项目将采集的 GUI 交互数据与通用多模态 SFT 数据集进行融合训练，总样本规模约为 600 万条。在 RFT 阶段，采用梯度正则化策略优化（Gradient-Regularized PolicyOptimization，GRPO）算法对模型进行强化学习，以增强其推理与思维能力。具体使用的文件参见下表。

| 文件名 | 格式 | 描述 |
| --- | --- | --- |
| finetune_ds.sh | shell | SFT命令脚本 |
| finetune.py | Python | SFT主程序 |
| dataset.py | Python | 构建dataset |
| trainer.py | Python | 构建trainer |
| fsdp.sh | shell | RFT命令脚本 |
| trainer/utils | 文件夹 | 被grpo.py调用 |
| fsdp2_dst.yml | YAML | RFT配置文件 |
| grpo.py | Python | RFT训练主程序 |
| trainer/arl.py | Python | 被grpo.py调用 |
| configs.py | Python | 被grpo.py调用 |

#### 评测相关文件

评估流程通过执行 eval.sh 脚本启动，首先由 run_predict_minicpm.py 自动完成模型推理，生成对应的预测结果；随后，该结果被传递至 run_eval_agent.py，进一步转换为标准化评估格式，并完成结果的汇总与指标计算，从而实现对模型性能的系统性评估。

| 文件名 | 格式 | 描述 |
| --- | --- | --- |
| eval.sh | shell | 推理评估脚本 |
| run_predict_minicpm.py | Python | 推理主程序 |
| run_eval_agent.py | Python | 评测程序 |
| utils | 文件夹 | 工具函数 |

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
