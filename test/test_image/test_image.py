import cv2

from detectnet_v2.inference_tlt import DetectNet
from argparser import parse_options


if __name__ == "__main__":
    parameters = parse_options()
    height = parameters['height_model']
    width = parameters['width_model']
    channel = parameters['channel']
    img_path = parameters['img_path']

    detectnet_obj = DetectNet(parameters['trt_path'], (channel, width, height),
                              parameters['num_class'],
                              parameters['batch_size'], parameters['box_norm'],
                              parameters['stride'])

    list_outputs = detectnet_obj.predict(img_path,
                                         parameters['confidence'],
                                         tuple(parameters['min_size']),
                                         parameters['nms_threshold'],
                                         parameters['scale_x'],
                                         parameters['scale_y'])

    print('\noutputs : {}'.format(list_outputs))
