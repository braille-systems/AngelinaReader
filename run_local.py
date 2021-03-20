#!/usr/bin/env python
# coding: utf-8
"""
Local application for Angelina Braille Reader inference
"""
import argparse
import os
from pathlib import Path

import local_config
import model.infer_retinanet as infer_retinanet

parser = argparse.ArgumentParser(description="Angelina Braille Reader: optical Braille text recognizer .")

parser.add_argument("input", type=str, help="File(s) to be processed: image, pdf or zip file or directory name")
parser.add_argument(
    "results_dir",
    nargs="?",
    type=str,
    help="(Optional) output directory. If not specified, results are placed at input location",
)
parser.add_argument(
    "-l", "--lang", type=str, default="RU", help="(Optional) Document language (RU or EN). If not specified, is RU"
)
parser.add_argument(
    "-o", "--orient", action="store_false", help="Don't find orientation, use original file orientation"
)
parser.add_argument("-2", dest="two", action="store_true", help="Process 2 sides")
parser.add_argument("--weights_file", type=str, default="weights/model.t7",
                    help=" (Optional) path to weights for RetinaNet")
parser.add_argument("--param_file", type=str, default="weights/param.txt",
                    help="(Optional) path to params for RetinaNet")
parser.add_argument("--save_dev", action="store_true", help="Save development info (pseudo-labels, log)")

args = parser.parse_args()

if not Path(args.input).exists():
    print("input file/path does not exist: " + args.input)
    exit()

recognizer = infer_retinanet.BrailleInference(
    params_fn=os.path.join(local_config.data_path, args.param_file),
    model_weights_fn=os.path.join(local_config.data_path, args.weights_file),
    create_script=None,
)

if Path(args.input).is_dir():
    results_dir = args.results_dir or args.input
    recognizer.process_dir_and_save(
        str(Path(args.input) / "**" / "*.*"),
        results_dir,
        lang=args.lang,
        extra_info=None,
        draw_refined=recognizer.DRAW_NONE,
        remove_labeled_from_filename=False,
        find_orientation=args.orient,
        align_results=True,
        process_2_sides=args.two,
        repeat_on_aligned=False,
        save_development_info=args.save_dev,
    )
else:
    results_dir = args.results_dir or Path(args.input).parent
    if Path(args.input).suffix == ".zip":
        recognizer.process_archive_and_save(
            args.input,
            results_dir,
            lang=args.lang,
            extra_info=None,
            draw_refined=recognizer.DRAW_NONE,
            remove_labeled_from_filename=False,
            find_orientation=args.orient,
            align_results=True,
            process_2_sides=args.two,
            repeat_on_aligned=False,
            save_development_info=args.save_dev,
        )
    elif Path(args.input).suffix.lower() in (".jpg", ".jpe", ".jpeg", ".png", ".gif", ".svg", ".bmp"):
        recognizer.run_and_save(
            args.input,
            results_dir,
            target_stem=None,
            lang=args.lang,
            extra_info=None,
            draw_refined=recognizer.DRAW_NONE,
            remove_labeled_from_filename=False,
            find_orientation=args.orient,
            align_results=True,
            process_2_sides=args.two,
            repeat_on_aligned=False,
            save_development_info=args.save_dev,
        )
    else:
        print("Incorrect file extention: " + Path(args.input).suffix + " . Only images, .pdf and .zip files allowed")
        exit()
print("Done. Results are saved in " + str(results_dir))
