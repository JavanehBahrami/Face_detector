from detectnet_v2.inference_tlt import DetectNet
import cv2
import time
import pickle
import sys
from epc.send_data import EPCMConfluentProducer

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-i', help='Input video path', required=True, type=str)
parser.add_argument('--model_path', help='TRT model path', required=True, type=str)

args = parser.parse_args()

video_path = args.i
trt_model_path = args.model_path

producer = EPCMConfluentProducer()

encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]


def filter_detection(detection_output):
	filtered = []
	for det in detection_output:
		if det['class_id'] == 0:
			filtered.append(det)
	return filtered

def prepare_data(frame, detection_out, stream_id, frame_id):
	result, encimg = cv2.imencode('.jpg', frame, encode_param)
	data = {
		'frame': encimg,
		'detection': detection_out,
		'stream_id': stream_id,
		'frame_id': frame_id
	}
	return data


def put_to_queue(producer, topic, data):
	producer.send_data(topic=topic, data=data)

counter = 0

detector = DetectNet(trt_model_path, input_size=(3, 736, 416), batch_size=1, num_class=1)
cap = cv2.VideoCapture(video_path)
while cap.isOpened():
    ret, frame = cap.read()
    st = time.time()
    det_out = detector.predict(frame, confidence=0.5)
    
    det_filtered = filter_detection(det_out)
    if det_filtered:
        data = prepare_data(frame, det_filtered, 10, counter)
        print(time.time()-st)
        put_to_queue(producer, 'detected_faces', data)
        
        print(counter)
    counter += 1
		
