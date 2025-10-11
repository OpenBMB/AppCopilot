---
license: apache-2.0
library_name: PaddleOCR
language:
- en
- zh
pipeline_tag: image-to-text
tags:
- OCR
- PaddlePaddle
- PaddleOCR
- textline_detection
---

# PP-OCRv5_server_det

## Introduction

PP-OCRv5_server_det is one of the PP-OCRv5_det series, the latest generation of text detection models developed by the PaddleOCR team. Designed for high-performance applications, it supports the detection of text in diverse scenarios—including handwriting, vertical, rotated, and curved text—across multiple languages such as Simplified Chinese, Traditional Chinese, English, and Japanese. Key features include robust handling of complex layouts, varying text sizes, and challenging backgrounds, making it suitable for practical applications like document analysis, license plate recognition, and scene text detection. The key accuracy metrics are as follow:

| Handwritten Chinese | Handwritten English | Printed Chinese | Printed English | Traditional Chinese | Ancient Text | Japanese | General Scenario | Pinyin | Rotation | Distortion | Artistic Text | Average | 
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.803 | 0.841 | 0.945	| 0.917	| 0.815 | 0.676 | 0.772	 | 0.797 | 0.671 | 0.8 | 0.876 | 0.673 | 0.827 |

## Quick Start

### Installation

1. PaddlePaddle

Please refer to the following commands to install PaddlePaddle using pip:

```bash
# for CUDA11.8
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# for CUDA12.6
python -m pip install paddlepaddle-gpu==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

# for CPU
python -m pip install paddlepaddle==3.0.0 -i https://www.paddlepaddle.org.cn/packages/stable/cpu/
```

For details about PaddlePaddle installation, please refer to the [PaddlePaddle official website](https://www.paddlepaddle.org.cn/en/install/quick).

2. PaddleOCR

Install the latest version of the PaddleOCR inference package from PyPI:

```bash
python -m pip install paddleocr
```

### Model Usage

You can quickly experience the functionality with a single command:

```bash
paddleocr text_detection \
    --model_name PP-OCRv5_server_det \
    -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png
```

You can also integrate the model inference of the text detection module into your project. Before running the following code, please download the sample image to your local machine.

```python
from paddleocr import TextDetection
model = TextDetection(model_name="PP-OCRv5_server_det")
output = model.predict(input="3ul2Rq4Sk5Cn-l69D695U.png", batch_size=1)
for res in output:
    res.print()
    res.save_to_img(save_path="./output/")
    res.save_to_json(save_path="./output/res.json")
```

After running, the obtained result is as follows:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'dt_polys': array([[[ 632, 1429],
        ...,
        [ 632, 1450]],

       ...,

       [[ 353,  102],
        ...,
        [ 353,  125]]], dtype=int16), 'dt_scores': [0.8436300312712586, 0.7779392262863483, ..., 0.8491056329808098]}}
```

The visualized image is as follows:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/rs6R4Ij5bfSeQ4mwLhcz6.jpeg)

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/module_usage/text_detection.html#iii-quick-start).

### Pipeline Usage

The ability of a single model is limited. But the pipeline consists of several models can provide more capacity to resolve difficult problems in real-world scenarios.

#### PP-OCRv5

The general OCR pipeline is used to solve text recognition tasks by extracting text information from images and outputting it in text form. And there are 5 modules in the pipeline: 
* Document Image Orientation Classification Module (Optional)
* Text Image Unwarping Module (Optional)
* Text Line Orientation Classification Module (Optional)
* Text Detection Module
* Text Recognition Module

Run a single command to quickly experience the OCR pipeline:

```bash
paddleocr ocr -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/3ul2Rq4Sk5Cn-l69D695U.png \
    --text_detection_model_name PP-OCRv5_server_det \
    --text_recognition_model_name PP-OCRv5_server_rec \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation True \
    --save_path ./output \
    --device gpu:0 
```

Results are printed to the terminal:

```json
{'res': {'input_path': '/root/.paddlex/predict_input/3ul2Rq4Sk5Cn-l69D695U.png', 'page_index': None, 'model_settings': {'use_doc_preprocessor': True, 'use_textline_orientation': True}, 'doc_preprocessor_res': {'input_path': None, 'page_index': None, 'model_settings': {'use_doc_orientation_classify': False, 'use_doc_unwarping': False}, 'angle': -1}, 'dt_polys': array([[[ 352,  105],
        ...,
        [ 352,  128]],

       ...,

       [[ 632, 1431],
        ...,
        [ 632, 1447]]], dtype=int16), 'text_det_params': {'limit_side_len': 64, 'limit_type': 'min', 'thresh': 0.3, 'max_side_limit': 4000, 'box_thresh': 0.6, 'unclip_ratio': 1.5}, 'text_type': 'general', 'textline_orientation_angles': array([0, ..., 0]), 'text_rec_score_thresh': 0.0, 'rec_texts': ['Algorithms for the Markov Entropy Decomposition', 'Andrew J. Ferris and David Poulin', 'Département de Physique, Université de Sherbrooke, Québec, JlK 2R1, Canada', '(Dated: October 31, 2018)', 'The Markov entropy decomposition (MED) is a recently-proposed, cluster-based simulation method for fi-', 'nite temperature quantum systems with arbitrary geometry. In this paper, we detail numerical algorithms for', 'performing the required steps of the MED, principally solving a minimization problem with a preconditioned', 'arXiv:1212.1442v1 [cond-mat.stat-mech] 6Dec 2012', "Newton's algorithm, as well as how to extract global susceptibilities and thermal responses. We demonstrate", 'the power of the method with the spin-1/2 XXZ model on the 2D square lattice, including the extraction of', 'critical points and details of each phase. Although the method shares some qualitative similarities with exact-', 'diagonalization, we show the MED is both more accurate and significantly more flexible.', 'PACS numbers: 05.10.−a,02.50.Ng, 03.67.−a,74.40.Kb', 'I.INTRODUCTION', 'This approximation becomes exact in the case of a 1D quan', 'tum (or classical) Markov chain [10], and leads to an expo-', 'Although the equations governing quantum many-body', 'nential reduction of cost for exact entropy calculations when', 'systems are simple to write down, finding solutions for the', 'the global density matrix is a higher-dimensional Markov net-', 'majority of systems remains incredibly difficult. Modern', 'work state [12, 13].', 'physics finds itself in need of new tools to compute the emer-', 'The second approximation used in the MED approach is', 'gent behavior of large, many-body systems.', 'related to the N-representibility problem. Given a set of lo-', 'There has been a great variety of tools developed to tackle', 'cal but overlapping reduced density matrices {pi}, it is a very', 'many-body problems, but in general, large 2D and 3D quan-', 'challenging problem to determine if there exists a global den-', 'tum systems remain hard to deal with. Most systems are', 'sity operator which is positive semi-definite and whose partial', 'thought to be non-integrable, so exact analytic solutions are', 'trace agrees with each ρi. This problem is QMA-hard (the', 'not usually expected. Direct numerical diagonalization can be', 'quantum analogue of NP) [14, 15], and is hopelessly diffi-', 'performed for relatively small systems — however the emer-', 'cult to enforce. Thus, the second approximation employed', 'gent behavior of a system in the thermodynamic limit may be', 'involves ignoring global consistency with a positive opera-', 'difficult to extract, especially in systems with large correlation', 'tor, while requiring local consistency on any overlapping re-', 'lengths. Monte Carlo approaches are technically exact (up to', 'gions between the ρi. At the zero-temperature limit, the MED', 'sampling error), but suffer from the so-called sign problem', 'approach becomes analogous to the variational nth-order re-', 'for fermionic, frustrated, or dynamical problems. Thus we are', 'duced density matrix approach, where positivity is enforced', 'limited to search for clever approximations to solve the ma-', 'on all reduced density matrices of size n [16–18].', 'jority of many-body problems.', 'The MED approach is an extremely flexible cluster method.', 'Over the past century, hundreds of such approximations', 'applicable to both translationally invariant systems of any di-', 'have been proposed, and we will mention just a few notable', 'mension in the thermodynamic limit, as well as finite systems', 'examples applicable to quantum lattice models. Mean-field', 'or systems without translational invariance (e.g. disordered', 'theory is simple and frequently arrives at the correct quali-', 'lattices, or harmonically trapped atoms in optical lattices).', 'tative description, but often fails when correlations are im-', 'The free energy given by MED is guaranteed to lower bound', 'portant. Density-matrix renormalisation group (DMRG) [1]', 'the true free energy, which in turn lower-bounds the ground', 'is efficient and extremely accurate at solving 1D problems,', 'state energy — thus providing a natural complement to varia-', 'but the computational cost grows exponentially with system', 'tional approaches which upper-bound the ground state energy.', 'size in two- or higher-dimensions [2, 3]. Related tensor-', 'The ability to provide a rigorous ground-state energy window', 'network techniques designed for 2D systems are still in their', 'is a powerful validation tool, creating a very compelling rea-', 'infancy [4–6].  Series-expansion methods [7] can be success-', 'son to use this approach.', 'ful, but may diverge or otherwise converge slowly, obscuring', 'In this paper we paper we present a pedagogical introduc-', 'the state in certain regimes. There exist a variety of cluster-', 'tion to MED, including numerical implementation issues and', 'based techniques, such as dynamical-mean-field theory [8]', 'applications to 2D quantum lattice models in the thermody-', 'and density-matrix embedding [9]', 'namiclimit.InSec.II.wegiveabriefderiyationofthe', 'Here we discuss the so-called Markov entropy decompo-', 'Markov entropy decomposition. Section III outlines a robust', 'sition (MED), recently proposed by Poulin & Hastings [10]', 'numerical strategy for optimizing the clusters that make up', '(and analogous to a slightly earlier classical algorithm [11]).', 'the decomposition. In Sec. IV we show how we can extend', 'This is a self-consistent cluster method for finite temperature', 'these algorithms to extract non-trivial information, such as', 'systems that takes advantage of an approximation of the (von', 'specific heat and susceptibilities. We present an application of', 'Neumann) entropy. In [10], it was shown that the entropy', 'the method to the spin-1/2 XXZ model on a 2D square lattice', 'per site can be rigorously upper bounded using only local in-', 'in Sec. V, describing how to characterize the phase diagram', 'formation — a local, reduced density matrix on N sites, say.', 'and determine critical points, before concluding in Sec. VI.'], 'rec_scores': array([0.99276221, ..., 0.95760632]), 'rec_polys': array([[[ 352,  105],
        ...,
        [ 352,  128]],

       ...,

       [[ 632, 1431],
        ...,
        [ 632, 1447]]], dtype=int16), 'rec_boxes': array([[ 352, ...,  128],
       ...,
       [ 632, ..., 1447]], dtype=int16)}}
```

If save_path is specified, the visualization results will be saved under `save_path`. The visualization output is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/SfWsVpad7eKD3I2uLYpnS.png)

The command-line method is for quick experience. For project integration, also only a few codes are needed as well:

```python
from paddleocr import PaddleOCR  

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_server_det",
    text_recognition_model_name="PP-OCRv5_server_rec",
    use_doc_orientation_classify=False, # Disables document orientation classification model via this parameter
    use_doc_unwarping=False, # Disables text image rectification model via this parameter
    use_textline_orientation=False, # Disables text line orientation classification model via this parameter
)
result = ocr.predict("./3ul2Rq4Sk5Cn-l69D695U.png")  
for res in result:  
    res.print()  
    res.save_to_img("output")  
    res.save_to_json("output")
```

For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/OCR.html#2-quick-start).

#### PP-StructureV3

Layout analysis is a technique used to extract structured information from document images. PP-StructureV3 includes the following six modules:
* Layout Detection Module
* General OCR Pipeline
* Document Image Preprocessing Pipeline （Optional）
* Table Recognition Pipeline （Optional）
* Seal Recognition Pipeline （Optional）
* Formula Recognition Pipeline （Optional）

Run a single command to quickly experience the PP-StructureV3 pipeline:

```bash
paddleocr pp_structurev3 -i https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/mG4tnwfrvECoFMu-S9mxo.png \
    --text_detection_model_name PP-OCRv5_server_det \
    --use_doc_orientation_classify False \
    --use_doc_unwarping False \
    --use_textline_orientation False \
    --device gpu:0
```

Results would be printed to the terminal. If save_path is specified, the results will be saved under `save_path`. The predicted markdown visualization is shown below:

![image/jpeg](https://cdn-uploads.huggingface.co/production/uploads/681c1ecd9539bdde5ae1733c/SfxF0X4drBTNGnfFOtZij.png)

Just a few lines of code can experience the inference of the pipeline. Taking the PP-StructureV3 pipeline as an example:

```python
from paddleocr import PPStructureV3

pipeline = PPStructureV3(
    text_detection_model_name="PP-OCRv5_server_det",
    use_doc_orientation_classify=False, # Use use_doc_orientation_classify to enable/disable document orientation classification model
    use_doc_unwarping=False,    # Use use_doc_unwarping to enable/disable document unwarping module
    use_textline_orientation=False, # Use use_textline_orientation to enable/disable textline orientation classification model
    device="gpu:0", # Use device to specify GPU for model inference
    )
output = pipeline.predict("./pp_structure_v3_demo.png")
for res in output:
    res.print() # Print the structured prediction output
    res.save_to_json(save_path="output") ## Save the current image's structured result in JSON format
    res.save_to_markdown(save_path="output") ## Save the current image's result in Markdown format
```

The default model used in pipeline is `PP-OCRv5_server_det`, and you can specify other text detection model by argument `text_detection_model_name`. And you can also use the local model file by argument `text_detection_model_dir`. For details about usage command and descriptions of parameters, please refer to the [Document](https://paddlepaddle.github.io/PaddleOCR/latest/en/version3.x/pipeline_usage/PP-StructureV3.html#2-quick-start).

## Links

[PaddleOCR Repo](https://github.com/paddlepaddle/paddleocr)

[PaddleOCR Documentation](https://paddlepaddle.github.io/PaddleOCR/latest/en/index.html)
