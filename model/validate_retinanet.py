#!/usr/bin/env python
# coding: utf-8
"""
evaluate levenshtein distance as recognition error for dataset using various model(s)
"""

# Для отладки
models = [
    # ('NN_saved/retina_chars_eced60', 'models/clr.008'),
    ('NN_results/all_data_0.5_100_5_nocls_91b802', 'models/clr.003.t7'),
    ('NN_results/all_data_0.5_100_5_nocls_91b802', 'models/clr.004.t7'),
    ('NN_results/all_data_0.5_100_5_nocls_91b802', 'models/clr.005.t7'),
    ('NN_results/all_data_0.5_100_5_nocls_91b802', 'models/clr.006.t7'),
]

model_dirs = [
    # 'NN_results/retina_chars_7e1d4e',
    # 'NN_results/retina_chars_785e35',
    # 'NN_results/retina_chars3_0.5_100_5_090399',
    # 'NN_results/retina_chars3_0.5_100_5_1c2265',
    # 'NN_results/retina_chars3_0.5_100_5_25be8f', ###
    ('NN_results/all_data_0.5_100_5_nocls_769014', 'models/clr.*.t7'),
    ('NN_results/retina_chars3_0.5_100_5_c3222a', 'models/clr.*.t7'),
    ('NN_results/retina_chars3_0.5_100_5_dirty_186ce5', 'models/clr.*.t7'),

    # ('NN_results/all_data_0.5_100_5_895449', 'models/clr.*.t7'),  ###
    # ('NN_results/all_data_0.5_100_5_7570a5', 'models/clr.*.t7'),  ###
    # ('NN_results/all_data_0.5_100_5_nocls_91b802', 'models/clr.*.t7'),  ###

    # 'NN_results/retina_chars_d58e5f',
    # 'NN_results/retina_chars_ff4ef7',
    # 'NN_results/retina_chars3_0.62_100_5_0473d8',
    # 'NN_results/retina_chars3_0.62_100_5_75df7f', ###--
    # 'NN_results/retina_chars3_0.62_100_5_fdb66e', ###
    #
    # 'NN_results/all_data_0.62_100_5_19c267',  ###
]

datasets = {
    # 'DSBI_train': [
    #                 r'DSBI\data\train.txt',
    #              ],
    # 'DSBI_test': [
    #                 r'DSBI\data\test.txt',
    #               ],
    'val_2': [
        r'My/labeled/labeled2/val_books.txt',
        r'My/labeled/labeled2/val_withtext.txt',
        r'My/labeled/labeled2/val_pupils.txt',
    ],
    'val_3_asi': [
        r'My/labeled/ASI/turlom_c15_photo_1p.txt',
    ],
    'val_3_asi_scan': [
        r'My/labeled/ASI/turlom_c15_scan_1p.txt',
    ],
}

lang = 'RU'

import os
import sys
import Levenshtein
from pathlib import Path
import PIL
import torch
sys.path.append(r'../..')
sys.path.append('../NN/RetinaNet')
import local_config
import data_utils.data as data
import data_utils.dsbi as dsbi
import braille_utils.postprocess as postprocess
import model.infer_retinanet as infer_retinanet
from braille_utils import label_tools

rect_margin=0.3

for md in model_dirs:
    models += [
        (str(md[0]), str(Path('models')/m.name))
        for m in (Path(local_config.data_path)/md[0]).glob(md[1])
    ]
for m in models:
    print(m)

def prepare_data():
    """
    data (datasets defined above as global) -> dict: key - list of dict (image_fn":full image filename, "gt_text": groundtruth pseudotext, "gt_rects": groundtruth rects + label 0..64)
    :return:
    """
    res_dict = dict()
    for key, list_file_names in datasets.items():
        data_list = list()
        res_dict[key] = data_list
        for list_file_name in list_file_names:
            list_file = os.path.join(local_config.data_path, list_file_name)
            data_dir = os.path.dirname(list_file)
            with open(list_file, 'r') as f:
                files = f.readlines()
            for fn in files:
                if fn[-1] == '\n':
                    fn = fn[:-1]
                full_fn = os.path.join(data_dir, fn)
                if os.path.isfile(full_fn):
                    rects = None
                    lbl_fn = full_fn.rsplit('.', 1)[0] + '.json'
                    if os.path.isfile(lbl_fn):
                        rects = data.read_LabelMe_annotation(label_filename=lbl_fn, get_points=False)
                    else:
                        lbl_fn = full_fn.rsplit('.', 1)[0] + '.txt'
                        if os.path.isfile(lbl_fn):
                            img = PIL.Image.open(full_fn)
                            rects = dsbi.read_DSBI_annotation(label_filename=lbl_fn,
                                                              width=img.width,
                                                              height=img.height,
                                                              rect_margin=rect_margin,
                                                              get_points=False
                                                              )
                        else:
                            full_fn = full_fn.rsplit('.', 1)[0] + '+recto.jpg'
                            lbl_fn  = full_fn.rsplit('.', 1)[0] + '.txt'
                            if os.path.isfile(lbl_fn):
                                img = PIL.Image.open(full_fn)
                                rects = dsbi.read_DSBI_annotation(label_filename=lbl_fn,
                                                                  width=img.width,
                                                                  height=img.height,
                                                                  rect_margin=rect_margin,
                                                                  get_points=False)
                    if rects is not None:
                        boxes = [r[:4] for r in rects]
                        labels = [r[4] for r in rects]
                        lines = postprocess.boxes_to_lines(boxes, labels, lang=lang)
                        gt_text = lines_to_pseudotext(lines)
                        data_list.append({"image_fn":full_fn, "gt_text": gt_text, "gt_rects": rects})
    return res_dict


def label_to_pseudochar(label):
    """
    int (0..63) - str ('0' .. 'o')
    """
    return chr(ord('0') + label)


def lines_to_pseudotext(lines):
    """
    lines (list of postprocess.Line) -> pseudotext (multiline '\n' delimetered, int labels converted to pdeudo chars)
    """
    out_text = []
    for ln in lines:
        if ln.has_space_before:
            out_text.append('')
        s = ''
        for ch in ln.chars:
            s += ' ' * ch.spaces_before + label_to_pseudochar(ch.label)
        out_text.append(s)
    return '\n'.join(out_text)


def pseudo_char_to_label010(ch):
    lbl = ord(ch) - ord('0')
    label_tools.validate_int(lbl)
    label010 = label_tools.int_to_label010(lbl)
    return label010


def count_dots_lbl(lbl):
    n = 0
    label010 = label_tools.int_to_label010(lbl)
    for c01 in label010:
        if c01 == '1':
            n += 1
        else:
            assert c01 == '0'
    return n

def count_dots_str(s):
    n = 0
    for ch in s:
        if ch in " \n":
            continue
        label010 = pseudo_char_to_label010(ch)
        for c01 in label010:
            if c01 == '1':
                n += 1
            else:
                assert c01 == '0'
    return n


def dot_metrics(res, gt):
    tp = 0
    fp = 0
    fn = 0
    opcodes = Levenshtein.opcodes(res, gt)
    for op, i1, i2, j1, j2 in opcodes:
        if op == 'delete':
            fp += count_dots_str(res[i1:i2])
        elif op == 'insert':
            fn += count_dots_str(gt[j1:j2])
        elif op == 'equal':
            tp += count_dots_str(res[i1:i2])
        elif op == 'replace':
            res_substr = res[i1:i2].replace(" ", "").replace("\n", "")
            gt_substr = gt[j1:j2].replace(" ", "").replace("\n", "")
            d = len(res_substr) - len(gt_substr)
            if d > 0:
                fp += count_dots_str(res_substr[-d:])
                res_substr = res_substr[:-d]
            elif d < 0:
                fn += count_dots_str(gt_substr[d:])
                gt_substr = gt_substr[:d]
            assert len(res_substr) == len(gt_substr)
            for i, res_i in enumerate(res_substr):
                res010 = pseudo_char_to_label010(res_i)
                gt010 = pseudo_char_to_label010(gt_substr[i])
                for p in range(6):
                    if res010[p] == '1' and gt010[p] == '0':
                        fp += 1
                    elif res010[p] == '0' and gt010[p] == '1':
                        fn += 1
                    elif res010[p] == '1' and gt010[p] == '1':
                        tp += 1
        else:
            raise Exception("incorrect operation " + op)

    return tp, fp, fn


def filter_lonely_rects(boxes, labels, img):
    dx_to_h = 2.35 # расстояние от края до центра 3го символа
    res_boxes = []
    res_labels = []
    filtered = []
    for i in range(len(boxes)):
        box = boxes[i]
        cy = (box[1] + box[3])/2
        dx = (box[3]-box[1])*dx_to_h
        for j in range(len(boxes)):
            if i == j:
                continue
            box2 = boxes[j]
            if (box2[0] < box[2] + dx) and (box2[2] > box[0] - dx) and (box2[1] < cy) and (box2[3] > cy):
                res_boxes.append(boxes[i])
                res_labels.append(labels[i])
                break
        else:
            filtered.append(box)
    if filtered:
        draw = PIL.ImageDraw.Draw(img)
        for b in filtered:
            draw.rectangle(b, fill="red")
        img.show()

    return res_boxes, res_labels

def dot_metrics_rects(boxes, labels, gt_rects, image_wh, img, do_filter_lonely_rects):
    if do_filter_lonely_rects:
        boxes, labels = filter_lonely_rects(boxes, labels, img)
    gt_labels = [r[4] for r in gt_rects]
    gt_rec_labels = [-1] * len(gt_rects)  # recognized label for gt, -1 - missed
    rec_is_false = [1] * len(labels)  # recognized is false

    if len(gt_rects) and len(labels):
        boxes = torch.tensor(boxes)
        gt_boxes = torch.tensor([r[:4] for r in gt_rects], dtype=torch.float32) * torch.tensor([image_wh[0], image_wh[1], image_wh[0], image_wh[1]])

        # Для отладки
        # labels = torch.tensor(labels)
        # gt_labels = torch.tensor(gt_labels)
        #
        # _, rec_order = torch.sort(boxes[:, 1], dim=0)
        # boxes = boxes[rec_order][:15]
        # labels = labels[rec_order][:15]
        # _, gt_order = torch.sort(gt_boxes[:, 1], dim=0)
        # gt_boxes = gt_boxes[gt_order][:15]
        # gt_labels = gt_labels[gt_order][:15]
        #
        # _, rec_order = torch.sort(labels, dim=0)
        # boxes = boxes[rec_order]
        # labels = labels[rec_order]
        # _, gt_order = torch.sort(-gt_labels, dim=0)
        # gt_boxes = gt_boxes[gt_order]
        # gt_labels = gt_labels[gt_order]
        #
        # labels = torch.tensor(labels)
        # gt_labels = torch.tensor(gt_labels)

        areas = (boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1])
        gt_areas = (gt_boxes[:, 2] - gt_boxes[:, 0])*(gt_boxes[:, 3] - gt_boxes[:, 1])
        x1 = torch.max(gt_boxes[:, 0].unsqueeze(1), boxes[:, 0].unsqueeze(0))
        y1 = torch.max(gt_boxes[:, 1].unsqueeze(1), boxes[:, 1].unsqueeze(0))
        x2 = torch.min(gt_boxes[:, 2].unsqueeze(1), boxes[:, 2].unsqueeze(0))
        y2 = torch.min(gt_boxes[:, 3].unsqueeze(1), boxes[:, 3].unsqueeze(0))
        intersect_area = (x2-x1).clamp(min=0)*(y2-y1).clamp(min=0)
        iou = intersect_area / (gt_areas.unsqueeze(1) + areas.unsqueeze(0) - intersect_area)
        for gt_i in range(len(gt_labels)):
            rec_i = iou[gt_i, :].argmax()
            if iou[gt_i, rec_i] > 0:
                gt_i2 = iou[:, rec_i].argmax()
                if gt_i2 == gt_i:
                    gt_rec_labels[gt_i] = labels[rec_i]
                    rec_is_false[rec_i] = 0

    tp = 0
    fp = 0
    fn = 0
    for gt_label, rec_label in zip(gt_labels, gt_rec_labels):
        if rec_label == -1:
            fn += count_dots_lbl(gt_label)
        else:
            res010 = label_tools.int_to_label010(rec_label)
            gt010 = label_tools.int_to_label010(gt_label)
            for p in range(6):
                if res010[p] == '1' and gt010[p] == '0':
                    fp += 1
                elif res010[p] == '0' and gt010[p] == '1':
                    fn += 1
                elif res010[p] == '1' and gt010[p] == '1':
                    tp += 1
    for label, is_false in zip(labels, rec_is_false):
        if is_false:
            fp += count_dots_lbl(label)
    return tp, fp, fn



def validate_model(recognizer, data_list, do_filter_lonely_rects):
    """
    :param recognizer: infer_retinanet.BrailleInference instance
    :param data_list:  list of (image filename, groundtruth pseudotext)
    :return: (<distance> avg. by documents, <distance> avg. by char, <<distance> avg. by char> avg. by documents>)
    """
    sum_d = 0
    sum_d1 = 0.
    sum_len = 0
    # по тексту
    tp = 0
    fp = 0
    fn = 0
    # по rect
    tp_r = 0
    fp_r = 0
    fn_r = 0

    for gt_dict in data_list:
        img_fn, gt_text, gt_rects = gt_dict['image_fn'], gt_dict['gt_text'], gt_dict['gt_rects']
        res_dict = recognizer.run(img_fn,
                                  lang=lang,
                                  draw_refined=infer_retinanet.BrailleInference.DRAW_NONE,
                                  find_orientation=False,
                                  process_2_sides=False,
                                  align_results=False,
                                  repeat_on_aligned=False,
                                  gt_rects=gt_rects)

        tpi, fpi, fni = dot_metrics_rects(boxes = res_dict['boxes'], labels = res_dict['labels'],
                                          gt_rects = res_dict['gt_rects'], image_wh = (res_dict['labeled_image'].width, res_dict['labeled_image'].height),
                                          img=res_dict['labeled_image'], do_filter_lonely_rects=do_filter_lonely_rects)
        tp_r += tpi
        fp_r += fpi
        fn_r += fni

        res_text = lines_to_pseudotext(res_dict['lines'])
        d = Levenshtein.distance(res_text, gt_text)
        sum_d += d
        if len(gt_text):
            sum_d1 += d/len(gt_text)
        sum_len += len(gt_text)
        tpi, fpi, fni = dot_metrics(res_text, gt_text)
        tp += tpi
        fp += fpi
        fn += fni

    # precision = tp/(tp+fp)
    # recall = tp/(tp+fn)
    precision_r = tp_r/(tp_r+fp_r) if tp_r+fp_r != 0 else 0.
    recall_r = tp_r/(tp_r+fn_r) if tp_r+fn_r != 0 else 0.
    return {
        # 'precision': precision,
        # 'recall': recall,
        # 'f1': 2*precision*recall/(precision+recall),
        'precision_r': precision_r,
        'recall_r': recall_r,
        'f1_r': 2*precision_r*recall_r/(precision_r+recall_r) if precision_r+recall_r != 0 else 0.,
        'd_by_doc': sum_d/len(data_list),
        'd_by_char': sum_d/sum_len,
        'd_by_char_avg': sum_d1/len(data_list)
    }


def main(table_like_format):
    verbose = 0
    # make data list
    data_set = prepare_data()
    prev_model_root = None
    do_filter_lonely_rects = False

    if table_like_format:
        print('model\tweights\tkey\t'
              'precision\trecall\tf1\t'
              'd_by_doc\td_by_char\td_by_char_avg')
    for model_root, model_weights in models:
        if model_root != prev_model_root:
            if not table_like_format:
                print('model: ', model_root)
            else:
                print()
            prev_model_root = model_root
        if verbose:
            print('evaluating weights: ', model_weights)
        params_fn = Path(local_config.data_path) / model_root / 'param.txt'
        if not params_fn.is_file():
            params_fn = Path(local_config.data_path) / (model_root + '.param.txt')  # старый вариант
            assert params_fn.is_file(), str(params_fn)
        recognizer = infer_retinanet.BrailleInference(
            params_fn=params_fn,
            model_weights_fn=os.path.join(local_config.data_path, model_root, model_weights),
            create_script=None,
            verbose=verbose)
        for key, data_list in data_set.items():
            res = validate_model(recognizer, data_list, do_filter_lonely_rects=do_filter_lonely_rects)
            # print('{model_weights} {key} precision: {res[precision]:.4}, recall: {res[recall]:.4} f1: {res[f1]:.4} '
            #       'precision_r: {res[precision_r]:.4}, recall_r: {res[recall_r]:.4} f1_r: {res[f1_r]:.4} '
            #       'd_by_doc: {res[d_by_doc]:.4} d_by_char: {res[d_by_char]:.4} '
            #       'd_by_char_avg: {res[d_by_char_avg]:.4}'.format(model_weights=model_weights, key=key, res=res))
            if table_like_format:
                print('{model}\t{weights}\t{key}\t'
                      '{res[precision_r]:.4}\t{res[recall_r]:.4}\t{res[f1_r]:.4}\t'
                      '{res[d_by_doc]:.4}\t{res[d_by_char]:.4}\t'
                      '{res[d_by_char_avg]:.4}'.format(model=model_root, weights=model_weights, key=key, res=res))
            else:
                print('{model_weights} {key} '
                      'precision_r: {res[precision_r]:.4}, recall_r: {res[recall_r]:.4} f1_r: {res[f1_r]:.4} '
                      'd_by_doc: {res[d_by_doc]:.4} d_by_char: {res[d_by_char]:.4} '
                      'd_by_char_avg: {res[d_by_char_avg]:.4}'.format(model_weights=model_weights, key=key, res=res))

if __name__ == '__main__':
    import time
    infer_retinanet.nms_thresh = 0.02
    postprocess.Line.LINE_THR = 0.6
    t0 = time.clock()
    # for thr in (0.5, 0.6, 0.7, 0.8):
    #     postprocess.Line.LINE_THR = thr
    #     print(thr)
    main(table_like_format=True)
    print(time.clock() - t0)
