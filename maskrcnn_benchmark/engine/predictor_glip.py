import cv2
import torch
import re
import numpy as np
from typing import List, Union
import nltk
import inflect
from transformers import AutoTokenizer
from torchvision import transforms as T
import pdb
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.structures.boxlist_ops import boxlist_iou
from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark import layers as L
from maskrcnn_benchmark.modeling.roi_heads.mask_head.inference import Masker
from maskrcnn_benchmark.utils import cv2_util

engine = inflect.engine()
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

import timeit

import sys


class GLIPDemo(object):
    def __init__(self,
                 cfg,
                 confidence_threshold=0.7,
                 min_image_size=None,
                 show_mask_heatmaps=False,
                 masks_per_dim=5,
                 load_model=True
                 ):
        self.cfg = cfg.clone()
        if load_model:
            self.model = build_detection_model(cfg)
            self.model.eval()
            self.device = torch.device(cfg.MODEL.DEVICE)
            self.model.to(self.device)
        self.min_image_size = min_image_size
        self.show_mask_heatmaps = show_mask_heatmaps
        self.masks_per_dim = masks_per_dim

        save_dir = cfg.OUTPUT_DIR
        if load_model:
            checkpointer = DetectronCheckpointer(cfg, self.model, save_dir=save_dir)
            _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()

        # used to make colors for each tokens
        mask_threshold = -1 if show_mask_heatmaps else 0.5
        self.masker = Masker(threshold=mask_threshold, padding=1)
        self.palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold

        self.tokenizer = self.build_tokenizer()

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg

        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size) if self.min_image_size is not None else lambda x: x,
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def build_tokenizer(self):
        cfg = self.cfg
        tokenizer = None
        if cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "bert-base-uncased":
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        elif cfg.MODEL.LANGUAGE_BACKBONE.TOKENIZER_TYPE == "clip":
            from transformers import CLIPTokenizerFast
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True, mask_token='ðŁĴĳ</w>')
            else:
                tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32",
                                                              from_slow=True)
        return tokenizer

    def run_ner(self, caption):
        noun_phrases = find_noun_phrases(caption)
        noun_phrases = [remove_punctuation(phrase) for phrase in noun_phrases]
        noun_phrases = [phrase for phrase in noun_phrases if phrase != '']
        relevant_phrases = noun_phrases
        labels = noun_phrases
        self.entities = labels

        tokens_positive = []

        for entity, label in zip(relevant_phrases, labels):
            try:
                # search all occurrences and mark them as different entities
                for m in re.finditer(entity, caption.lower()):
                    tokens_positive.append([[m.start(), m.end()]])
            except:
                print("noun entities:", noun_phrases)
                print("entity:", entity)
                print("caption:", caption.lower())

        return tokens_positive

    def inference(self, original_image, original_caption):
        predictions = self.compute_prediction(original_image, original_caption)
        top_predictions = self._post_process_fixed_thresh(predictions)
        return top_predictions

    def run_on_web_image(self, 
            original_image, 
            original_caption, 
            thresh=0.5,
            custom_entity = None,
            alpha = 0.0):
        predictions = self.compute_prediction(original_image, original_caption, custom_entity)
        top_predictions = self._post_process(predictions, thresh)

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions)
        result = self.overlay_boxes(result, top_predictions)
        result = self.overlay_entity_names(result, top_predictions)
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def visualize_with_predictions(self, 
            original_image, 
            predictions, 
            thresh=0.5,
            alpha=0.0,
            box_pixel=3,
            text_size = 1,
            text_pixel = 2,
            text_offset = 10,
            text_offset_original = 4,
            color = 255):
        
        # 获取所有唯一的标签
        labels = predictions.get_field("labels")
        unique_labels = torch.unique(labels)
        
        # 使用HSV颜色空间自动生成颜色
        self.label_colors = {}
        for i, label in enumerate(unique_labels):
            # 使用黄金比例来生成均匀分布的色调
            hue = (i * 137.5) % 360  # 137.5是黄金角度
            saturation = 95  # 高饱和度
            value = 95      # 高亮度
            
            # 转换HSV到BGR
            hsv_color = np.array([[[hue, saturation, value]]], dtype=np.float32)
            hsv_color = hsv_color.astype(np.uint8)
            bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0]
            
            # 存储颜色映射
            self.label_colors[label.item()] = bgr_color.tolist()

        height, width = original_image.shape[:-1]
        predictions = predictions.resize((width, height))
        top_predictions = self._post_process(predictions, thresh) # 根据阈值筛选预测结果

        result = original_image.copy()
        if self.show_mask_heatmaps:
            return self.create_mask_montage(result, top_predictions) # 根据配置选择可视化方式

        # result = self.overlay_boxes(result, top_predictions)
        # result = self.overlay_entity_names(result, top_predictions)
        # 绘制边界框
        result = self.overlay_boxes(result, top_predictions, alpha=alpha, box_pixel=box_pixel)
        # 添加实体名称
        result = self.overlay_entity_names(result, top_predictions, text_size=text_size, text_pixel=text_pixel, text_offset = text_offset, text_offset_original = text_offset_original)
        
        if self.cfg.MODEL.MASK_ON:
            result = self.overlay_mask(result, top_predictions)
        return result, top_predictions

    def compute_prediction(self, original_image, original_caption, custom_entity = None):
        # image
        image = self.transforms(original_image)
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # caption
        if isinstance(original_caption, list):
            # we directly provided a list of category names
            caption_string = ""
            tokens_positive = []
            seperation_tokens = " . "
            for word in original_caption:
                
                tokens_positive.append([[len(caption_string), len(caption_string) + len(word)]])
                caption_string += word
                caption_string += seperation_tokens
            
            tokenized = self.tokenizer([caption_string], return_tensors="pt")

            original_caption = caption_string
            print(tokens_positive)
        else:
            tokenized = self.tokenizer([original_caption], return_tensors="pt")
            if custom_entity is None:
                tokens_positive = self.run_ner(original_caption)
            else:
                tokens_positive = []
                for entity in custom_entity:
                    if "char_bounds" in entity:
                        # char bounds provided
                        tokens_positive.append([list(entity["char_bounds"])])
                    else:
                        # online look for char bounds
                        for i, m in enumerate(re.finditer(entity["span"], original_caption)):
                            if i >= 1:
                                print(
                                    f"More than 1 match with the phrase {entity['span']} in {entity['span']}! We only take the first match."
                                )
                                break
                            tokens_positive.append([[m.start(), m.end()]])
            print(tokens_positive)
        # process positive map
        positive_map = create_positive_map(tokenized, tokens_positive)

        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0

        positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=plus)
        self.plus = plus
        self.positive_map_label_to_token = positive_map_label_to_token
        tic = timeit.time.perf_counter()

        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list, captions=[original_caption], positive_map=positive_map_label_to_token)
            predictions = [o.to(self.cpu_device) for o in predictions]
        print("inference time per image: {}".format(timeit.time.perf_counter() - tic))

        # always single image is passed at a time
        prediction = predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        prediction = prediction.resize((width, height))

        if prediction.has_field("mask"):
            # if we have masks, paste the masks in the right position
            # in the image, as defined by the bounding boxes
            masks = prediction.get_field("mask")
            # always single image is passed at a time
            masks = self.masker([masks], [prediction])[0]
            prediction.add_field("mask", masks)

        return prediction

    def _post_process_fixed_thresh(self, predictions):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = self.confidence_threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = self.confidence_threshold[0]
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1)
        predictions = predictions[keep]

        scores = predictions.get_field("scores")
        _, idx = scores.sort(0, descending=True)
        return predictions[idx]

    def _post_process(self, predictions, threshold=0.5):
        scores = predictions.get_field("scores")
        labels = predictions.get_field("labels").tolist()
        thresh = scores.clone()
        for i, lb in enumerate(labels):
            if isinstance(self.confidence_threshold, float):
                thresh[i] = threshold
            elif len(self.confidence_threshold) == 1:
                thresh[i] = threshold
            else:
                thresh[i] = self.confidence_threshold[lb - 1]
        keep = torch.nonzero(scores > thresh).squeeze(1) # 筛选高于阈值的预测结果
        predictions = predictions[keep]

        scores = predictions.get_field("scores") # 获取筛选后的置信度分数
        _, idx = scores.sort(0, descending=True) # 按置信度降序排序
        return predictions[idx]

    def compute_colors_for_labels(self, labels):
        # """
        # Simple function that adds fixed colors depending on the class
        # """
        # colors = (30 * (labels[:, None] - 1) + 1) * self.palette
        # colors = (colors % 255).numpy().astype("uint8")
        # try:
        #     colors = (colors * 0 + self.color).astype("uint8")
        # except:
        #     pass
        # return colors

        # """
        # 使用HSV颜色空间生成不同的颜色
        # """
        # num_labels = len(labels)
        # hsv_colors = []
        # for i, label in enumerate(labels):
        #     # 使用标签值来生成不同的色调
        #     hue = (label * 137.5) % 360  # 使用黄金角度137.5来生成分散的色调
        #     saturation = 100
        #     value = 100
            
        #     # 转换HSV到RGB
        #     hsv_color = np.array([[[hue, saturation, value]]], dtype=np.float32)
        #     rgb_color = cv2.cvtColor(hsv_color.astype(np.uint8), cv2.COLOR_HSV2RGB)[0][0]
        #     hsv_colors.append(rgb_color)
        
        # return np.array(hsv_colors, dtype=np.uint8)

        """
        为不同的类别生成不同的颜色
        """
        # # 定义一组鲜艳的颜色 (BGR格式)
        # colors = [
        #     [255, 0, 0],    # 蓝色
        #     [0, 255, 0],    # 绿色
        #     [0, 0, 255],    # 红色
        #     [255, 255, 0],  # 青色
        #     [255, 0, 255],  # 粉色
        #     [0, 255, 255],  # 黄色
        #     [128, 0, 0],    # 深蓝
        #     [0, 128, 0],    # 深绿
        #     [0, 0, 128],    # 深红
        #     [128, 128, 0],  # 深青
        #     [128, 0, 128],  # 深粉
        #     [0, 128, 128],  # 深黄
        # ]
        
        # 为每个标签分配颜色
        label_colors = []
        for label in labels:
            color_idx = (label.item() - 1) % len(self.colors)  # 循环使用颜色列表
            label_colors.append(self.colors[color_idx])
        
        return np.array(label_colors, dtype=np.uint8)

    def overlay_boxes(self, image, predictions, alpha=0.5, box_pixel = 3):
        labels = predictions.get_field("labels")
        boxes = predictions.bbox

        # colors = self.compute_colors_for_labels(labels).tolist() # 为不同标签计算不同的颜色
        new_image = image.copy()
        for box, label in zip(boxes, labels):
            box = box.to(torch.int64)  # 将边界框坐标转换为整数
            top_left, bottom_right = box[:2].tolist(), box[2:].tolist() # 获取左上角和右下角坐标
            color = self.label_colors[label.item()]
            new_image = cv2.rectangle( # 使用OpenCV绘制矩形
                new_image, tuple(top_left), tuple(bottom_right), tuple(color), box_pixel)

        # Following line overlays transparent rectangle over the image
        image = cv2.addWeighted(new_image, alpha, image, 1 - alpha, 0) # 使用透明度混合原始图像和带有边界框的图像

        return image

    def overlay_scores(self, image, predictions):
        scores = predictions.get_field("scores")
        boxes = predictions.bbox

        for box, score in zip(boxes, scores):
            box = box.to(torch.int64)
            image = cv2.putText(image, '%.3f' % score,
                                (int(box[0]), int((box[1] + box[3]) / 2)),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

        return image

    def overlay_entity_names(self, image, predictions, names=None, text_size=1.0, text_pixel=2, text_offset = 10, text_offset_original = 4):
        scores = predictions.get_field("scores").tolist()
        labels = predictions.get_field("labels").tolist()
        original_labels = predictions.get_field("labels").tolist()  # 保存原始的标签索引
        new_labels = []
        if self.cfg.MODEL.RPN_ARCHITECTURE == "VLDYHEAD":
            plus = 1
        else:
            plus = 0
        self.plus = plus
        if self.entities and self.plus: # 根据实体列表生成标签
            for i in labels:
                if i <= len(self.entities):
                    new_labels.append(self.entities[i - self.plus])
                else:
                    new_labels.append('object')
            # labels = [self.entities[i - self.plus] for i in labels ]
        else:
            new_labels = ['object' for i in labels]
        boxes = predictions.bbox

        template = "{}:{:.2f}"  # 设置文本模板
        previous_locations = []
        for box, score, orig_label, label in zip(boxes, scores, original_labels, new_labels): # 为每个检测框添加文本
            x, y = box[:2] 
            s = template.format(label, score).replace("_", " ").replace("(", "").replace(")", "") # 格式化文本，移除特殊字符
            for x_prev, y_prev in previous_locations: # 处理文本重叠
                if abs(x - x_prev) < abs(text_offset) and abs(y - y_prev) < abs(text_offset):
                    y -= text_offset
                    
            # 使用与边界框相同的颜色
            color = self.label_colors[orig_label]
            cv2.putText( # 绘制文本
                image, s, (int(x), int(y)-text_offset_original), cv2.FONT_HERSHEY_SIMPLEX, text_size, color, text_pixel, cv2.LINE_AA
            )
            previous_locations.append((int(x), int(y))) # 记录文本位置


        return image

    def overlay_mask(self, image, predictions):
        masks = predictions.get_field("mask").numpy()
        labels = predictions.get_field("labels")

        colors = self.compute_colors_for_labels(labels).tolist()

        # import pdb
        # pdb.set_trace()
        # masks = masks > 0.1

        for mask, color in zip(masks, colors):
            thresh = mask[0, :, :, None].astype(np.uint8)
            contours, hierarchy = cv2_util.findContours(
                thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
            )
            image = cv2.drawContours(image, contours, -1, color, 2)

        composite = image

        return composite

    def create_mask_montage(self, image, predictions):
        masks = predictions.get_field("mask")
        masks_per_dim = self.masks_per_dim
        masks = L.interpolate( # 调整mask大小
            masks.float(), scale_factor=1 / masks_per_dim
        ).byte()
        height, width = masks.shape[-2:] # 获取调整后的mask尺寸
        max_masks = masks_per_dim ** 2 # 计算最大mask数量
        masks = masks[:max_masks] # 限制mask数量
        # handle case where we have less detections than max_masks
        if len(masks) < max_masks: # 处理mask数量不足的情况
            masks_padded = torch.zeros(max_masks, 1, height, width, dtype=torch.uint8) # 创建填充的零矩阵
            masks_padded[: len(masks)] = masks # 将实际mask填充到零矩阵中
            masks = masks_padded
        masks = masks.reshape(masks_per_dim, masks_per_dim, height, width) # 重塑mask数组为网格形状
        result = torch.zeros( # 创建结果图像
            (masks_per_dim * height, masks_per_dim * width), dtype=torch.uint8
        )
        for y in range(masks_per_dim): # 将mask放置到结果图像中
            start_y = y * height
            end_y = (y + 1) * height
            for x in range(masks_per_dim):
                start_x = x * width
                end_x = (x + 1) * width
                result[start_y:end_y, start_x:end_x] = masks[y, x]

        return cv2.applyColorMap(result.numpy(), cv2.COLORMAP_JET), None # 应用热力图颜色映射


def create_positive_map_label_to_token_from_positive_map(positive_map, plus=0):
    positive_map_label_to_token = {}
    for i in range(len(positive_map)):
        positive_map_label_to_token[i + plus] = torch.nonzero(positive_map[i], as_tuple=True)[0].tolist()
    return positive_map_label_to_token


def create_positive_map(tokenized, tokens_positive):
    """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
    positive_map = torch.zeros((len(tokens_positive), 256), dtype=torch.float)

    for j, tok_list in enumerate(tokens_positive):
        for (beg, end) in tok_list:
            try:
                beg_pos = tokenized.char_to_token(beg)
                end_pos = tokenized.char_to_token(end - 1)
            except Exception as e:
                print("beg:", beg, "end:", end)
                print("token_positive:", tokens_positive)
                # print("beg_pos:", beg_pos, "end_pos:", end_pos)
                raise e
            if beg_pos is None:
                try:
                    beg_pos = tokenized.char_to_token(beg + 1)
                    if beg_pos is None:
                        beg_pos = tokenized.char_to_token(beg + 2)
                except:
                    beg_pos = None
            if end_pos is None:
                try:
                    end_pos = tokenized.char_to_token(end - 2)
                    if end_pos is None:
                        end_pos = tokenized.char_to_token(end - 3)
                except:
                    end_pos = None
            if beg_pos is None or end_pos is None:
                continue

            assert beg_pos is not None and end_pos is not None
            positive_map[j, beg_pos: end_pos + 1].fill_(1)
    return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)


def find_noun_phrases(caption: str) -> List[str]:
    caption = caption.lower()
    tokens = nltk.word_tokenize(caption)
    pos_tags = nltk.pos_tag(tokens)

    grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(pos_tags)

    noun_phrases = list()
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            noun_phrases.append(' '.join(t[0] for t in subtree.leaves()))

    return noun_phrases


def remove_punctuation(text: str) -> str:
    punct = ['|', ':', ';', '@', '(', ')', '[', ']', '{', '}', '^',
             '\'', '\"', '’', '`', '?', '$', '%', '#', '!', '&', '*', '+', ',', '.'
             ]
    for p in punct:
        text = text.replace(p, '')
    return text.strip()
