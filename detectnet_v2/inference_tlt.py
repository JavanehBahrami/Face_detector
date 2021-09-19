import cv2
import pycuda
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
import numpy as np


class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class DetectNet(object):
    def __init__(self, trt_path, input_size=(3, 960, 544), num_class=3,
                 batch_size=1, box_norm=35.0, stride=16):
        self.trt_path = trt_path
        self.input_size = input_size
        self.batch_size = batch_size
        self.box_norm = box_norm
        self.stride = stride
        self.num_class = num_class
        self.list_output = []

        (
            self.grid_w,
            self.grid_h,
            self.grid_size,
            self.grid_centers
        ) = self._compute_grids()

        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_engine = self._load_engine(trt_runtime, self.trt_path)
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers()

        self.context = self.trt_engine.create_execution_context()

    def _load_engine(self, trt_runtime, engine_path):
        with open(engine_path, "rb") as f:
            engine_data = f.read()
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def _allocate_buffers(self):
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        binding_to_type = {
            "input_1": np.float32,
            "output_bbox/BiasAdd": np.float32,
            "output_cov/Sigmoid": np.float32}

        for binding in self.trt_engine:
            size = trt.volume(self.trt_engine.get_binding_shape(binding)) \
                   * self.batch_size
            dtype = binding_to_type[str(binding)]
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))
            if self.trt_engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def _do_inference(self, context, bindings, inputs,
                      outputs, stream):
        [cuda.memcpy_htod_async(inp.device, inp.host, stream)
         for inp in inputs]
        context.execute_async(
            batch_size=self.batch_size, bindings=bindings,
            stream_handle=stream.handle)

        [cuda.memcpy_dtoh_async(out.host, out.device, stream)
         for out in outputs]

        stream.synchronize()

        return [out.host for out in outputs]

    def _process_image(self, img, w, h):
        image_resized = cv2.resize(img, (w, h))
        img_np = image_resized.transpose((2, 0, 1))
        img_np = img_np.ravel()
        img_np = img_np.astype(np.float32) / 255

        return img_np

    def _check_bbox(self, bbox, score, min_size, w_img, h_img):
        w_min_box = min_size[0]
        h_min_box = min_size[1]
        # new_bboxes, new_scores = [], []
        new_bbox, new_score = [], -1

        # for box_id, bbox in enumerate(bboxes):
        bbox[0] = max(0, bbox[0])
        bbox[1] = max(0, bbox[1])
        bbox[2] = min(w_img, bbox[2])
        bbox[3] = min(h_img, bbox[3])
        w_box = bbox[2]
        h_box = bbox[3]
        if w_box > w_min_box and h_box > h_min_box:
            new_bbox = bbox
            new_score = score

        return new_bbox, new_score

    def _rescale_image_bbox(self, bbox, w_img, h_img):
        x_scale = w_img / self.input_size[1]
        y_scale = h_img / self.input_size[2]

        x_min = round(int(bbox[0]) * x_scale)
        y_min = round(int(bbox[1]) * y_scale)
        x_max = round(int(bbox[2] + bbox[0]) * x_scale)
        y_max = round(int(bbox[3] + bbox[1]) * y_scale)

        rescale_img_bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

        return rescale_img_bbox

    @staticmethod
    def _scale_xy_box(box, image_width, image_height, scale_x=1.0,
                      scale_y=1.0):
        x_0 = box[0]
        y_0 = box[1]
        x_1 = box[0]+box[2]
        y_1 = box[1]+box[3]

        width = box[2]
        height = box[3]

        width_scale = ((scale_x - 1) / 2) * width
        height_scale = ((scale_y - 1) / 2) * height

        x_0 = int(max(x_0 - width_scale, 0))
        y_0 = int(max(y_0 - height_scale, 0))
        x_1 = int(min(x_1 + width_scale, image_width))
        y_1 = int(min(y_1 + height_scale, image_height))

        scaled_box = [
            x_0, y_0, x_1-x_0, y_1-y_0
        ]

        return scaled_box

    def predict(self, image, confidence=0.1, min_size=(50, 50),
                nms_threshold=0.5, scale_x=1.0, scale_y=1.0,
                bbox_mode=None):

        self.list_output.clear()
        w, h = self.input_size[1], self.input_size[2]
        h_img, w_img, _ = image.shape
        img = self._process_image(image, w, h)

        np.copyto(self.inputs[0].host, img)

        detection_boxes, detection_scores = self._do_inference(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        bboxes, class_ids, scores = self._postprocess(
            detection_boxes,
            detection_scores,
            confidence,
            tuple(range(self.num_class)),
        )

        box_indexes = cv2.dnn.NMSBoxes(bboxes, scores,
                                       confidence, nms_threshold)

        for idx in box_indexes:
            idx = int(idx)
            rescale_box = self._rescale_image_bbox(bboxes[idx], w_img,
                                                   h_img)
            scaled_box = self._scale_xy_box(rescale_box, w_img, h_img,
                                            scale_x, scale_y)
            new_bbox, new_score = self._check_bbox(scaled_box, scores[idx],
                                                   min_size, w_img, h_img)
            if len(new_bbox) > 0 and new_score > 0:
                score = np.float32(new_score)

                res = {"class_id": class_ids[idx],
                       "confidence": round(np.float32(score).item(), 2),
                       "bounding_box": new_bbox}
                self.list_output.append(res)

        if bbox_mode == 'percentage':
            for output in self.list_output:
                bbox = output['bounding_box']
                x_min = np.round(bbox[0]/w_img, 3)
                y_min = np.round(bbox[1]/h_img, 3)
                w_box = np.round(bbox[2]/w_img, 3)
                h_box = np.round(bbox[3]/h_img, 3)
                up_dict = {"bounding_box": [x_min, y_min, w_box, h_box]}
                output.update(up_dict)

        return self.list_output

    def _compute_grids(self):
        grid_h = int(self.input_size[2] / self.stride)
        grid_w = int(self.input_size[1] / self.stride)

        grid_size = grid_h * grid_w

        grid_centers_w = []
        grid_centers_h = []

        for i in range(grid_h):
            value = (i * self.stride + 0.5) / self.box_norm
            grid_centers_h.append(value)

        for i in range(grid_w):
            value = (i * self.stride + 0.5) / self.box_norm
            grid_centers_w.append(value)

        grid_centers_w = np.array(grid_centers_w, dtype=np.float32)
        grid_centers_h = np.array(grid_centers_h, dtype=np.float32)
        new_grid_centers = np.empty((grid_h, grid_w, 2), dtype=np.float32)

        new_grid_center_w = [grid_centers_w] * len(grid_centers_h)
        new_grid_center_h = [grid_centers_h] * len(grid_centers_w)
        new_grid_centers[..., 0] = np.array(new_grid_center_h).T
        new_grid_centers[..., 1] = np.array(new_grid_center_w)

        return grid_w, grid_h, grid_size, new_grid_centers

    def _postprocess(self, detection_boxes, detection_scores, min_confidence,
                     analysis_classes, wh_format=True):
        bbs = []
        class_ids = []
        scores = []

        for c in analysis_classes:
            x1_idx = c * 4 * self.grid_size
            y1_idx = x1_idx + self.grid_size
            x2_idx = y1_idx + self.grid_size
            y2_idx = x2_idx + self.grid_size

            boxes = detection_boxes

            h, w = np.mgrid[0:self.grid_h, 0:self.grid_w]

            i = w + (h * self.grid_w)
            flat_i = i.ravel()

            low_bound = c * self.grid_size
            high_bound = (c + 1) * self.grid_size

            score_val = detection_scores[low_bound:high_bound]

            new_bbox_idx = flat_i
            o1 = boxes[x1_idx + new_bbox_idx]

            o2 = boxes[y1_idx + new_bbox_idx]
            o3 = boxes[x2_idx + new_bbox_idx]
            o4 = boxes[y2_idx + new_bbox_idx]

            flat_center_w = self.grid_centers[..., 1].ravel()
            flat_center_h = self.grid_centers[..., 0].ravel()
            o1 = (o1 - flat_center_w) * -self.box_norm
            o2 = (o2 - flat_center_h) * -self.box_norm
            o3 = (o3 + flat_center_w) * self.box_norm
            o4 = (o4 + flat_center_h) * self.box_norm

            filter_score_idx = np.where(score_val > min_confidence)

            new_o1 = o1[filter_score_idx]
            new_o2 = o2[filter_score_idx]
            new_o3 = o3[filter_score_idx]
            new_o4 = o4[filter_score_idx]

            xmin = np.int_(new_o1)
            ymin = np.int_(new_o2)
            xmax = np.int_(new_o3)
            ymax = np.int_(new_o4)

            bbs_np = np.zeros((len(new_o1), 4), dtype=int)
            if wh_format:
                bbs_np[..., 0] = xmin
                bbs_np[..., 1] = ymin
                bbs_np[..., 2] = xmax - xmin
                bbs_np[..., 3] = ymax - ymin
                bbs_np.tolist()
                bbs.extend(bbs_np)

            class_ids_np = np.repeat(c, len(new_o1))
            class_ids.extend(class_ids_np.tolist())

            selected_score = score_val[filter_score_idx]
            scores.extend((np.float32(selected_score)).tolist())

        return bbs, class_ids, scores
