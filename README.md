# PeopleNet tlt model based on DetectNet v2
These models accept 960x544x3 dimension input tensors and outputs 60x34x12 bbox coordinate tensor and 60x34x3 class confidence tensor.

url : https://ngc.nvidia.com/catalog/models/nvidia:tlt_peoplenet

In this code we aimed to inference from a tensorrt model which is obtained from a tlt model and test the outputs.

To this end, you need to convert your tlt model into trt by using two commands:
1. tlt-export to convert your tlt model into etlt
2. tlt-converter to convert your etlt model into trt

# multiple batch
url : https://github.com/NVIDIA/DL4AGX/blob/master/MultiDeviceInferencePipeline/training/objectDetection/ssdConvertUFF/utils/inference.py

## Requirements
tlt container (version >= 3) which supports TensorRT and pycuda
1. tensorrt version : 7.2.1.6
2. cuda : cuda-11.1

An etlt model:
tlt-export detectnet_v2 -m model_input.tlt \
                -k tlt_key \
                -o model_output.etlt \
                -e spec_files/train_spec_file.txt \
                --batch_size 1 \
                --data_type fp32

A trt model:
tlt-converter -k tlt_key \
               -d channel,height,width \
               -o NMS \
               -e model_output.trt `Your trt model name` \
               -m 1 \
               -t fp16 \
               -i nchw \
               model_input.etlt  `Your etlt model name`

## Running the code
<br>for running the model, one can easily run the `python3 example.py` the container:
>python example.py


## Parameters in the example file:

1. input of the example code is an image.
2. the output of the example code is a list of dictionaries which includes 3 options:
`class_id`, `confidence` and `bounding_box`

[{'class_id': 2, 'confidence': 1.0, 'bounding_box': [0.629, 0.13, 0.088, 0.169]}

class_id values are like this:

|   class_id  |    category  |
|:------------| :-----------:|
|      0      |     person   |
|      1      |     bag      |
|      2      |     face     |

the format of bounding box output is like this(`percentage of width_image and heigh_image`):
[x_min, y_min, w_bbox, h_bbox]