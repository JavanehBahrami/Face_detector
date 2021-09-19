# Unit Test
In this test we want to test the detector module with different image inputs.
image input type can be one of this:
1. `image`
2. `zero` image
3. `one` image
4. `random` image
5. None image


## Requirements
tlt container (version == 3) which has: TensorRT


## Running the code
<br>for running the model, one can easily run the `python3 test_image.py` with this argument in the container:
>python test_detector.py -c parameter_config.json


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
11. trt_path : A path of your TensorRT detector model, (string value)
12. img_path : A path of your image, (string value)

