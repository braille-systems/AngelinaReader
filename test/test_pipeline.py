from pathlib import Path

import PIL
import numpy as np
import torch
from ovotools import AttrDict

import braille_utils.label_tools as lt
from braille_utils import postprocess
from data_utils.data import ImagePreprocessor, unify_shape, AugMode
from model.infer_retinanet import BrailleInferenceImpl, BrailleInference, OrientationAttempts


def test_inference_pipeline():  # testing the whole pipeline
    recognizer = BrailleInference(params_fn="../weights/param.txt",
                                  model_weights_fn="../weights/model.t7")
    out_dir = Path("./out")  # TODO add environmental variable
    out_dir.mkdir(exist_ok=True)
    recognizer.run_and_save(
        img="../brl_ocr/data/labeled/books/golubina/IMG_2354.JPG",
        results_dir=out_dir,
        target_stem=None,
        lang="RU",
        extra_info=None,
        draw_refined=recognizer.DRAW_NONE,
        remove_labeled_from_filename=False,
        find_orientation=True,
        align_results=True,
        process_2_sides=False,
        repeat_on_aligned=False,
        save_development_info=False,
    )


def test_recognizer_pipeline():  # testing code from recognizer's __init__() and run_and_save()
    # a possible scenario for `__init__()`
    params = AttrDict.load(params_fn="../weights/param.txt")  # TODO create issue in OvoTools
    inference_width = 850
    params.data.net_hw = (inference_width, inference_width)  # TODO what is it? Input image width & height?
    params.data.batch_size = 1
    params.augmentation = AttrDict(
        img_width_range=(inference_width, inference_width),
        stretch_limit=0.0,
        rotate_limit=0,
    )
    preprocessor = ImagePreprocessor(params, mode=AugMode.inference.value)
    model_weights_fn = "../weights/model.t7"
    device = "cuda:0"
    impl = BrailleInferenceImpl(params, model_weights_fn, device=device, label_is_valid=lt.label_is_valid)
    impl.to(device)

    # a possible scenario for `run_and_save()`
    # entering `run()`
    img = PIL.Image.open("./data/test_recognizer_input.jpg")

    np_img = np.asarray(img)
    aug_img, aug_gt_rects = preprocessor.preprocess_and_augment(np_img)
    aug_img = unify_shape(aug_img)
    input_tensor = preprocessor.to_normalized_tensor(aug_img, device=impl.device)
    input_tensor_rotated = torch.tensor(0).to(impl.device)
    # TODO find orientation
    with torch.no_grad():
        boxes, labels, scores, best_idx, err_score, boxes2, labels2, scores2 = impl(
            input_tensor, input_tensor_rotated, find_orientation=False, process_2_sides=False
        )
    boxes = boxes.tolist()
    labels = labels.tolist()
    scores = scores.tolist()
    min_align_score = 0

    lines = postprocess.boxes_to_lines(
        boxes,
        labels,
        scores=scores,
        lang="RU",
        filter_lonely=False,  # = SAVE_FOR_PSEUDOLABELS_MODE != 1,
        min_align_score=min_align_score,
    )
    BrailleInference.refine_lines(lines)

    aug_img = PIL.Image.fromarray(aug_img if best_idx < OrientationAttempts.ROT90 else None)
    if best_idx in (OrientationAttempts.ROT180, OrientationAttempts.ROT270):
        aug_img = aug_img.transpose(PIL.Image.ROTATE_180)
    # TODO check homography
    results_dict = {
        "image": aug_img,
        "best_idx": best_idx,
        "err_scores": list([ten.cpu().data.tolist() for ten in err_score]),
        "gt_rects": aug_gt_rects,
        "homography": None,
    }
    # postprocess.pseudolabeling_spellchecker(lines, to_score=pseudolabel_scores[0]) # TODO pseudolabeling
    results_dict.update(
        BrailleInference.draw_results(aug_img, boxes, lines, labels, scores, reverse_page=False, draw_refined=0))
    out_dir = Path("./out")
    out_dir.mkdir(exist_ok=True)
    BrailleInference.save_results(results_dict, reverse_page=False, results_dir=out_dir,
                                  filename_stem="test_recognizer",
                                  save_development_info=True)


if __name__ == "__main__":  # TODO find out why pytest doesn't recognize tests
    test_inference_pipeline()
    test_recognizer_pipeline()
