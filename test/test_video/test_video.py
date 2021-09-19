import cv2
import os
import numpy as np
import torch
import time
import sys

from detector import DetectNet


def read_frames(video_path, skip_frame=None):
    clip = []
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i in range(0, num_frames):
        ret, frame = cap.read()
        if ret:
            frame_ch = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if skip_frame is not None and i % skip_frame == 0:
                clip.append(frame_ch)
            elif skip_frame is None:
                clip.append(frame_ch)

    return clip, width, height, fps


def filter_outputs(outputs_list):
    bboxes, confidences, class_ids, detected_faces = [], [], [], []

    for item in outputs_list:
        class_id = item['class_id']
        if class_id == 2:
            bbox = item['bounding_box']
            bboxes.append(bbox)

            confidence = item['confidence']
            confidences.append(confidence)

            class_ids.append(class_id)
            face_bbox = (bbox[0], bbox[1], bbox[2], bbox[3])
            detected_faces.append(face_bbox)

    return bboxes, confidences, class_ids, detected_faces


def draw_bboxes(imgid, image, class_id_list, confidence_list, bbox_list):
    bboxes, confidences, class_ids = [], [], []

    for bbox, conf, cls_id in zip(bbox_list, confidence_list, class_id_list):
        bboxes.append(bbox)
        confidences.append(conf)
        class_ids.append(cls_id)

    clr = (0, 0, 255)
    for bb, conf, cls_id in zip(bboxes, confidences, class_ids):
        cv2.rectangle(image, (bb[0], bb[1]),
                      (bb[0] + bb[2], bb[1] + bb[3]), clr, 2)

        label = 'conf : '+str(conf)+"  _  cls_id : "+str(cls_id)

        (label_width, label_height), baseLine = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

        y_label = max(bb[1], label_height)

        cv2.rectangle(image, (bb[0], y_label - label_height),
                      (bb[0] + label_width, y_label + baseLine),
                      (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (bb[0], y_label),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
        cv2.putText(image, str(imgid), (40, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3)

    return image


def write_video(out, frame):
    out.write(frame)

    return


def process_img(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image


def process_video(input_video_path, output_video_path, model):
    frames_list, w, h, fps = read_frames(input_video_path)

    cap = cv2.VideoCapture(input_video_path)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_draw_list = []

    out = cv2.VideoWriter(output_video_path, fourcc, fps,
                          (w, h), True)

    for imgid, image in enumerate(frames_list[:]):
        outputs_list = model.predict(image, 0.5, (50, 50), 0.6, 1.3, 1.3)
        bbox_list, confidence_list, class_id_list, detected_faces = \
            filter_outputs(outputs_list)
        print('image_id = ', imgid)
        image = process_img(image)

        image = draw_bboxes(imgid, image, class_id_list, confidence_list, bbox_list)

        write_video(out, image)

    return


if __name__ == '__main__':

    abs_path = '/workspace'
    input_video_path = os.path.join(abs_path, "iou_tracker/my_iou_refactored"
                                    "/facerec_test3.mp4")

    model_path = '/workspace/iou_tracker/resources/model'
    trt_path = os.path.join(model_path, 'detector', 'tlt3',
                            "resnet34_peoplenet_pruned-quantized_b1_int8.trt")

    width = 960
    height = 544
    num_class = 3
    batch_size = 1
    box_norm = 35.0
    stride = 16
    model = DetectNet(trt_path, (3, width, height),
                      num_class, batch_size, box_norm, stride)

    output_video_path = os.path.join(abs_path,
                                     "python_inference/detection_output/"
                                     "facerec_test3_out.avi")

    process_video(input_video_path, output_video_path, model)
