#  file: evaluate.py
import argparse
import json
import logging
import os

import cv2
import numpy as np

try:
    import rasterio
except ImportError:
    rasterio = None

from quality_metrics import metric_functions

logger = logging.getLogger(__name__)


def read_image(path):
    logger.info("Reading image %s", os.path.basename(path))
    if rasterio and (path.endswith(".tif") or path.endswith(".tiff")):
        return np.rollaxis(rasterio.open(path).read(), 0, 3)
    return cv2.imread(path)


def evaluation(org_img_path, pred_img_path, metrics):
    output_dict = {}
    org_img = read_image(org_img_path)
    pred_img = read_image(pred_img_path)

    for metric in metrics:
        metric_func = metric_functions[metric]
        out_value = float(metric_func(org_img, pred_img))
        logger.info(f"{metric.upper()} value is: {out_value}")
        output_dict[metric] = out_value
    return output_dict

def rename2PicsToSameNames(path1, path2):
    pics1 = os.listdir(path1)
    pics2 = os.listdir(path2)

    sum = 0

    for pic in pics1:
        print(pic)
        try:
            flag = pics2.index(pic.replace('_real_B', '_fake_B'))
        except IOError:
            print('There is not any image with the same name as ' + str(pic) + ' in ' + str(path2))
        p1_name = "{:0>6d}".format(sum) + pic[-4:]
        p2_name = p1_name
        src_p1 = os.path.join(path1, pic)
        dst_p1 = os.path.join(path1, p1_name)
        os.rename(src_p1, dst_p1)
        print(str(src_p1) + ' --> ' + str(dst_p1) + '(renamed)')

        src_p2 = os.path.join(path2, pic.replace('_real_B', '_fake_B'))
        dst_p2 = os.path.join(path2, p2_name)
        os.rename(src_p2, dst_p2)
        print(str(src_p2) + ' --> ' + str(dst_p2) + '(renamed)')
        sum += 1

def main():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    all_metrics = sorted(metric_functions.keys())
    parser = argparse.ArgumentParser(description="Evaluates an Image Super Resolution Model")
    parser.add_argument("--org_img_path", help="Path to original input image", required=True, metavar="FOLDER")
    parser.add_argument("--pred_img_path", help="Path to predicted image", required=True, metavar="FOLDER")
    parser.add_argument("--metric", dest="metrics", action="append",
                        choices=all_metrics + ['all'], metavar="METRIC",
                        help="select an evaluation metric (%(choices)s) (can be repeated)")
    args = parser.parse_args()
    if not args.metrics:
        args.metrics = ["psnr"]
    if "all" in args.metrics:
        args.metrics = all_metrics

    path1 = args.org_img_path
    path2 = args.pred_img_path

    # rename2PicsToSameNames(path1, path2)

    pics1 = os.listdir(path1)

    tot_ssim = 0.0
    tot_psnr = 0.0
    tot = 0

    parser.set_defaults(metrics=['ssim'])
    args = parser.parse_args()

    for pic in pics1:
        p1 = os.path.join(path1, pic)
        p2 = os.path.join(path2, pic)
        result_dict = evaluation(p1, p2, args.metrics)
        print(result_dict['ssim'])

        tot_ssim += result_dict['ssim']

        tot += 1

    parser.set_defaults(metrics=['psnr'])

    for pic in pics1:
        p1 = os.path.join(path1, pic)
        p2 = os.path.join(path2, pic)
        result_dict = evaluation(p1, p2, args.metrics)

        print(result_dict['psnr'])

        tot_psnr += result_dict['psnr']
        tot += 1


    print('***************')
    #print(tot)
    print('Average SSIM: ' + str(tot_ssim / tot / 2)) # origin: tot_ssim / tot
    print('Average PSNR: ' + str(tot_psnr / tot / 2)) # origin: tot_psnr / tot


if __name__ == "__main__":
    main()
