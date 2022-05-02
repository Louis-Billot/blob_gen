# Instructions for the installation and use of the .BLOB file generator

The source of this procedure is [this Collab notebook](https://colab.research.google.com/github/luxonis/depthai-ml-training/blob/master/colab-notebooks/YoloV5_training.ipynb) where the authors explain the training of a custom network before sending it to a DepthAI camera.

> This procedure is written for Ubuntu but easily works inside a WSL distro on Windows.

## Table of Content

- [Setup](#setup-of-all-the-required-tools)
- [Usage](#usage)
- [ONNX generation](#optional---onnx-file-from-a-yolov5-model)
- [On-Device Decoding](#optional---pruning-an-onnx-model-for-on-device-decoding)

## Setup of all the required tools

Create a working directory and download the installation files for the OpenVINO toolkit (here version 2021.4 LTS)

```bash
mkdir -p blob_gen/openvino_installer
cd blob_gen/openvino_installer/
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18319/l_openvino_toolkit_p_2021.4.752.tgz
```

Extract the files and install the dependencies
```bash
tar xf l_openvino_toolkit_p_2021.4.752.tgz
cd l_openvino_toolkit_p_2021.4.752/
sudo ./install_openvino_dependencies.sh
```

Accept the license terms and install the toolkit
```bash
sed -i.bak 's/decline/accept/g' silent.cfg
sudo ./install.sh --silent silent.cfg
```

Finally, install the prerequisites for the model optimizer tool
```bash
bash /opt/intel/openvino_2021/deployment_tools/model_optimizer/install_prerequisites/install_prerequisites.sh
```

## Usage

An .ONNX file is required to perform the following commands, see [`this example for generating such a file`](#optional---onnx-file-from-a-yolov5-model).

Setup the environment
```bash
cd blob_gen
source /opt/intel/openvino_2021/bin/setupvars.sh
```

Optimize the ONNX model to create an OpenVINO model (.bin & .xml files)
```bash
mo.py --data_type FP16 --reverse_input_channel --scale 255 --output_dir openvino_models/ --input_model ../yolov5/pretrained/yolov5n.onnx
```
> Adding the input shape to the parameters may remove some errors `--input_shape [1,3,640,640]`

> If you [pruned the ONNX model](#optional---pruning-an-onnx-model-for-on-device-decoding) this parameter is required `--output "output1_yolov5,output2_yolov5,output3_yolov5"`

Compile the OpenVINO model for your device specifically
```bash
mkdir openvino_models/blobs
/opt/intel/openvino_2021.4.752/deployment_tools/tools/compile_tool/compile_tool -ip U8 -d MYRIAD -VPU_NUMBER_OF_SHAVES 4 -VPU_NUMBER_OF_CMX_SLICES 4 -m openvino_models/yolov5n.xml -o openvino_models/blobs/yolov5n_640_fp16_u8_4.blob
```
> -d specifies the processor to compile for and -ip allows to set the input precision (here unsigned 8bits integer)

> SHAVES is the name of the Intel VPU cores. Specify the number you want and provide the same value for the CMX 

If you executed all this in a WSL distro you can export the .BLOB file back to windows like this:
```bash
mkdir /mnt/c/depthai_blob_files
cp -i openvino_models/blobs/yolov5n_640_fp16_u8_4.blob /mnt/c/depthai_blob_files/
```

## OPTIONAL - ONNX file from a YOLOv5 model
The following commands are an example to convert a pretrained YOLOv5 network to ONNX
> The model used here is the YOLOv5 nano trained on the COCO dataset
```bash
cd ~
git clone https://github.com/ultralytics/yolov5.git
pip install -U -r yolov5/requirements.txt
mkdir -p yolov5/pretrained
cd yolov5/pretrained/
wget https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5n.pt

cd ~
python yolov5/export.py --img 640 --batch 1 --device cpu --include "onnx" --simplify --weights yolov5/pretrained/yolov5n.pt
```
> The `--img` parameter is crucial to correctly define !!

## OPTIONAL - Pruning an ONNX model for on-device decoding

This is an example on how to modify the neural network to add on-device decoding.
> This is based on the YOLOv5 architecture and modifications may be required for application to another architecture.

Install the ONNX python library::
```bash
pip install onnx
```

Define the paths to the ONNX file to modify and to the resulting file before executing this python script:
```python
onnx_path = '/content/yolov5/yolov5m.onnx'
output_path = '/content/yolov5/pruned.onnx'

import onnx

onnx_model = onnx.load(onnx_path)

conv_indices = []
for i, n in enumerate(onnx_model.graph.node):
  if "Conv" in n.name:
    conv_indices.append(i)

input1, input2, input3 = conv_indices[-3:]

sigmoid1 = onnx.helper.make_node(
    'Sigmoid',
    inputs=[onnx_model.graph.node[input1].output[0]],
    outputs=['output1_yolov5'],
)

sigmoid2 = onnx.helper.make_node(
    'Sigmoid',
    inputs=[onnx_model.graph.node[input2].output[0]],
    outputs=['output2_yolov5'],
)

sigmoid3 = onnx.helper.make_node(
    'Sigmoid',
    inputs=[onnx_model.graph.node[input3].output[0]],
    outputs=['output3_yolov5'],
)

onnx_model.graph.node.append(sigmoid1)
onnx_model.graph.node.append(sigmoid2)
onnx_model.graph.node.append(sigmoid3)
onnx.save(onnx_model, output_path)
```

If you open the resulting file in [Netron](https://netron.app/), you should see 3 sigmoids added right before the post-processing