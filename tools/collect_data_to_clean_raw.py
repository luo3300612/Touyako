import os.path
from os.path import join as opj
import lmdb
import numpy as np
import cv2
import argparse
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import tarfile
import shutil
from glob import glob
from PIL import Image
from touyako.config import upload_root


class ParallelElements:
    # 并行Elements
    # 元素级的并行
    # 支持tqdm
    # TODO 支持无output
    # TODO 支持无多进程debug
    def __init__(self, processes):
        self.processes = processes
        self.pool = Pool(processes=processes)

    def run(self, func, args: list):
        pbar = tqdm(total=len(args))
        update = lambda *args: pbar.update()

        for i, arg in enumerate(args):
            self.pool.apply_async(func, (arg,), callback=update)
        self.pool.close()
        self.pool.join()
        # res = []
        # for output in outputs:
        #     res.append(output.get())
        # return res


def get_image_key(entry):
    return entry.split('|')[-1].split('\t')[0]


def load_sub_image_raw(entry):
    add_bbox = False
    if '#' in entry:
        image_path, *bbox = entry.split('#')
        x1 = int(bbox[-4])
        y1 = int(bbox[-3])
        x2 = int(bbox[-2])
        y2 = int(bbox[-1])
        add_bbox = True
    else:
        image_path = entry

    img = cv2.imread(image_path)
    if img is None:
        return None
    if add_bbox:
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)

    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def load_and_save(entry, save_folder):
    entry, *labels = entry.strip().split('\t')
    md5 = entry.split('/')[-1]

    if '#' in md5:  # is sub image
        md5 = md5.replace('.', 'DOT')
        md5 = md5 + '.jpg'

    save_path = opj(save_folder, md5)
    # print('save to', save_path)
    if args.direct_copy:  # 直接拷贝，避免多次jpg保存导致的图片质量下降
        shutil.copy(entry, save_path)
    else:
        img = load_sub_image_raw(entry)
        if img is None:
            return
        cv2.imwrite(save_path, img)


def load_and_save_for_pe(entry):
    return load_and_save(entry, args.output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, default='input txt ann infos')
    parser.add_argument('--output', type=str, default='output image folder')
    parser.add_argument('--compress', action='store_true', default=False)
    parser.add_argument('--direct_copy', action='store_true', default=False)
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--upload', action='store_true', default=False)
    parser.add_argument('--upload_folder', type=str, default='')
    parser.add_argument('--debug', action='store_true', default=False)
    args = parser.parse_args()

    # if args.upload is True:
    #     assert args.compress is True, "you must compress before upload"
    #     assert args.upload_folder != '', "please specify upload target"

    setattr(args, 'output', os.path.abspath(args.output))

    if os.path.exists(args.output):
        c = input(f'output folder {args.output} exists rm it?(y)')
        if c != 'y':
            exit(0)
        shutil.rmtree(args.output)

    os.mkdir(args.output)

    infos = []
    with open(args.input) as f:
        for line in f:
            infos.append(line.split('\t')[0])

    if args.debug:
        print("Debug mode ON")
        infos = infos[:5]

    print('extracting images...')
    # for info in tqdm(infos):
    #     load_and_save_for_pe(info)
    pe = ParallelElements(16)
    pe.run(load_and_save_for_pe, infos)

    if args.compress:
        print('compressing...')
        os.system(f'cd {args.output} && cd .. && tar -cvf {os.path.basename(args.output)}.tar {args.output}')

    n_input = len(infos)
    n_succeed = len(glob(opj(args.output, '*')))
    print(f'N input: {n_input}, N succeed: {n_succeed}, {n_input / n_succeed * 100:.2f}')

    if args.upload:
        os.system(f'cd {args.output} && cd .. && ytdp_upload {args.output}.tgz {opj(upload_root, args.upload_folder)}')
