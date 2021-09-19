import cv2
import numpy as np

from detectnet_v2.inference_tlt import DetectNet
from argparser import parse_options


def generate_inputs(image_path, input_type='image', w=408, h=612, ch=3):
    if input_type == 'image':
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif input_type == 'zero':
        image = np.zeros((w, h, ch))
    elif input_type == 'one':
        image = np.ones((w, h, ch))
    elif input_type == 'random':
        image = np.random.standard_normal([w, h, ch]) * 255
        image = image.astype(np.uint8)
    elif input_type is None:
        image = None

    return image


if __name__ == "__main__":
    parameters = parse_options()
    height = parameters['height_model']
    width = parameters['width_model']
    channel = parameters['channel']
    img_path = parameters['img_path']

    # input_type can be {'image', 'zero', 'one', 'random', None}
    image = generate_inputs(img_path, input_type='random')

    detectnet_obj = DetectNet(parameters['trt_path'], (channel, width, height),
                              parameters['num_class'],
                              parameters['batch_size'], parameters['box_norm'],
                              parameters['stride'])

    list_outputs = detectnet_obj.predict(image,
                                         parameters['confidence'],
                                         tuple(parameters['min_size']),
                                         parameters['nms_threshold'],
                                         parameters['scale_x'],
                                         parameters['scale_y'])

    print('\noutputs : {}'.format(list_outputs))
