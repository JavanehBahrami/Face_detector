import cv2
import os

from detectnet_v2.inference_tlt import DetectNet
from argparser import parse_options


def visualize_outputs(img_path, output_list, counter):
    img = cv2.imread(img_path)
    image = img
    height, width,_ = img.shape

    for item in output_list:
        bbox = item['bounding_box']
        color = [255, 0, 0]
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    cv2.imwrite('batch_infr_people_'+str(counter)+'.jpg', image)

    return



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

    outputs_list = detectnet_obj.predict(img_path,
                                         parameters['confidence'],
                                         tuple(parameters['min_size']),
                                         parameters['nms_threshold'],
                                         parameters['scale_x'],
                                         parameters['scale_y'])

    if parameters['mode_visualize'] == True:
        img_list = os.listdir(img_path)  
        for idx, image_output in enumerate(outputs_list):
            print('\nimage_id : {}, outputs : {}'.format(idx+1, image_output))
            image_path = os.path.join(img_path, img_list[idx])
            visualize_outputs(image_path, image_output, idx)
    else:
        for idx, image_output in enumerate(outputs_list):
            print('\nimage_id : {}, outputs : {}'.format(idx+1, image_output))
