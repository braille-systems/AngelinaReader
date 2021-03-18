import random
from pathlib import Path

import PIL
import numpy as np
from ovotools import AttrDict

from braille_utils import postprocess
from data_utils.data import read_labelme_annotation, ImagePreprocessor, AugMode, unify_shape
from model.infer_retinanet import OrientationAttempts, BrailleInference


def albumentations_to_labelme(x: float, img_length: int) -> float:  # here `img_length` may be height or width of img
    return x * img_length


def test_albumentations():
    params = AttrDict.load(params_fn="../weights/param.txt")
    inference_width = 850
    params.data.net_hw = (inference_width, inference_width)  # TODO what is it? Input image width & height?
    params.data.batch_size = 1
    params.augmentation = AttrDict(
        img_width_range=(inference_width, inference_width),
        stretch_limit=0.1,
        rotate_limit=5,
        ShiftScaleRotate=True,
        Perspective=True,
        Affine=True
    )
    img = PIL.Image.open("./data/test_albumentations_input.jpg")
    bboxes = read_labelme_annotation("./data/test_albumentations_input.json")
    np_img = np.asarray(img)

    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)

    random.seed(0)
    for run_idx in range(5):
        preprocessor = ImagePreprocessor(params, mode=AugMode.train.value)
        aug_img, aug_gt_rects = preprocessor.preprocess_and_augment(np_img, rects=bboxes)
        aug_img = unify_shape(aug_img)
        aug_img = PIL.Image.fromarray(aug_img)
        results_dict = {
            "image": aug_img,
            "best_idx": OrientationAttempts.NONE,
            "err_scores": [],
            "gt_rects": aug_gt_rects,
            "homography": None,
        }

        aug_width, aug_height = aug_img.size

        aug_bboxes = [(albumentations_to_labelme(aug_gt_rect[0], aug_width),
                       albumentations_to_labelme(aug_gt_rect[1], aug_height),
                       albumentations_to_labelme(aug_gt_rect[2], aug_width),
                       albumentations_to_labelme(aug_gt_rect[3], aug_height)) for aug_gt_rect in aug_gt_rects]
        labels = [rect[4] for rect in aug_gt_rects]
        scores = [rect[5] for rect in aug_gt_rects]  # should be an array of ones
        lines = postprocess.boxes_to_lines(
            aug_bboxes,
            labels=labels,
            scores=scores,
            lang="RU",
            filter_lonely=False,  # as if SAVE_FOR_PSEUDOLABELS_MODE != 1,
            min_align_score=0,
        )
        BrailleInference.refine_lines(lines)
        reverse_page = False
        results_dict.update(
            BrailleInference.draw_results(aug_img=aug_img, boxes=aug_bboxes, lines=lines, labels=labels, scores=scores,
                                          reverse_page=reverse_page, draw_refined=0)
        )
        BrailleInference.save_results(results_dict, reverse_page=reverse_page, results_dir=out_dir,
                                      filename_stem="test_augmentations_{}".format(run_idx),
                                      save_development_info=True)


if __name__ == "__main__":
    test_albumentations()
