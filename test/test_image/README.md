# test trt model for a single image
In this code we aimed to inference from a tensorrt model which is obtained from a tlt model and test the outputs.
To this end, you need to convert your tlt model into trt by using two commands:
1. tlt-export to convert your tlt model into etlt
2. tlt-converter to convert your etlt model into trt


## Requirements
tlt container (version >= 2) which has: TensorRT
An etlt model:

tlt-export detectnet_v2 -m model_input.tlt \
                -k tlt_key \
                -o model_output.etlt \
                -e spec_files/train_spec_file.txt \
                --batch_size 1 \
                --data_type fp32

A trt model (batch_size = 1 for a single image):

tlt-converter -k tlt_key \
               -d channel,height,width \
               -o NMS \
               -e model_output.trt `Your trt model name` \
               -m 1 \
               -t fp16 \
               -i nchw \
               model_input.etlt  `Your etlt model name`

## Running the code
<br>for running the model, one can easily run the `python3 test_image.py` with this argument in the container:
>python test_image.py -c parameter_config.json


## Parameters in the config file:

1. width_model : Width size of model for peoplenet it should be 960,
2. height_model : Height size of model for peoplenet it should be 544,
3. batch_size : Batch size of trt model, Default is 1,
4. stride : Stride number of shifting detected bounding boxes to the predifined grid cells, Default is 16 (int value),
5. box_norm : Box norm value of shifting detected bounding boxes to the predifined grid cells, Default is 35.0 (float value),
6. num_class : Number of classes, Default is 2,
7. confidence : Threshold fo minimum confidence, Default is 0.5 (float value),
8. min_size" : Threshold for Minimum size of detected bboxes, Default is [30, 30] (tuple of int value),
9. nms_threshold : Threshold fo minimum NMS filter, Default is 0.5 (float value),
10. margin_pad : Margin value to add padding to the detected bboxes in order to regulize them, Default is [10, 10] (tuple of int value),
11. trt_path : A path of your TensorRT model, Default is "/workspace/test_tlt/model/resnet34_peoplenet_pruned_b1_int8.trt" (string value),
12. img_path : A path of your, Default is "./people.jpg",

