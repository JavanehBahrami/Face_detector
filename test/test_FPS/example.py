import cv2
import time
import os

from detectnet_v2.inference_tlt import DetectNet


def set_parameters(model_type):
    if model_type == "detectnet_v2":
        height = 544
        width = 960
        channel = 3
        model_path = '/workspace/iou_tracker/resources/model'
        trt_path = os.path.join(model_path, 'detector', 'tlt3',
                                "resnet34_peoplenet_pruned-quantized_b1_int8.trt")

        num_class = 3
        batch_size = 1
        box_norm = 35.0
        stride = 16

    return (channel, width, height), trt_path, num_class, \
        batch_size, box_norm, stride


if __name__ == "__main__":
    image_path = "/workspace/python_inference/input_image/img.jpg"

    confidence = 0.5
    min_size = (50, 50)
    nms_threshold = 0.5
    scale_x = 1.3
    scale_y = 1.2

    model_type = "detectnet_v2"

    model_size, trt_path, num_class, batch_size, box_norm, stride = \
        set_parameters(model_type)

    detectnet_obj = DetectNet(trt_path, model_size, num_class,
                              batch_size, box_norm, stride)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    list_outputs = detectnet_obj.predict(image, confidence, min_size,
                                         nms_threshold, scale_x, scale_y)
    start = time.time()
    overal_time = 0
    counter = 1000
    for i in range(counter):
        start = time.time()
        list_outputs = detectnet_obj.predict(image, confidence, min_size,
                                             nms_threshold, scale_x, scale_y)
        end = time.time()
        overal_time += end - start
    print('mean overal time = ', overal_time / counter)
    print('process time = ', end - start)
    print('\noutputs : {}'.format(list_outputs))
