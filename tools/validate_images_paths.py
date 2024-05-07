import hashlib
import os
import os.path
from os.path import join as opj
from PIL import Image
from multiprocessing import Pool, cpu_count
import math
from tqdm import tqdm
import time
import random
import imagehash
import argparse
import pickle as pkl
from collections import defaultdict
import shutil
import PIL
from pathlib import Path

from touyako.image_dedup_tools import *
from touyako.file_utils import txt_loader, get_image_paths_under_folder, valid_image_exts, ann_loader
from touyako.image_utils import get_phash, get_md5
from touyako.mp_tools import ParallelElements


# 验证image_paths.txt文件，输出一个去重后的image_paths.txt
def get_image_info(image_path):
    try:
        image = Image.open(image_path)
        h, w = image.size
        md5 = get_md5(image)
        phash = get_phash(image).hash
    # except (IOError, FileNotFoundError, PIL.UnidentifiedImageError) as e:
    except (IOError, OSError, FileNotFoundError, PIL.Image.DecompressionBombError) as e:
        md5 = -1
        phash = -1
        h = -1
        w = -1
        return -1, -1, (-1, -1)
    try:
        image.seek(1)
    except EOFError:  # 确认是图片
        return md5, phash, (h, w)
    except ValueError:  # 是gif动图
        return -1, -1, (-1, -1)
    return -1, -1, (-1, -1)


def check_file(path):
    if os.path.exists(path):  # 防止覆盖
        print(f'output file {path} exists! overwrite? (y?)')
        c = input()
        if c != 'y':
            exit(0)

    filename = path.split('/')[-1]
    folder = path[:-len(filename)]

    if not os.path.exists(folder):
        print(f'Folder {folder}, not exists!')
        exit(0)


def format_output(info):
    if args.rename_image:
        source_path = Path(os.path.abspath(info['path']))
        suffix = source_path.suffix
        md5 = info['md5']
        md5_filename = f'{md5}{suffix}'
        target_path = source_path.parent / md5_filename
        os.rename(str(source_path), str(target_path))  # 比shutil.move快
    else:
        target_path = Path(os.path.abspath(info['path']))

    if len(info['label']) != 0:
        ann = f'{str(target_path)}\t{info["label"]}\n'
    else:
        ann = str(target_path) + '\n'
    return ann


if __name__ == '__main__':
    # print("CPU的核数为：{}".format(cpu_count()))
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input image path file')
    parser.add_argument('--output', type=str, help='output info file', default='')
    parser.add_argument('--n-pools', type=int, default=16, help='number of max pools')
    parser.add_argument('--phash-thresh', type=int, default=3, help='thresh for phash dedup')
    parser.add_argument('--rename_image', action='store_true', default=False, help='rename image to its md5')
    parser.add_argument('--label', type=str, default='', help='rename image to its md5')
    parser.add_argument('--save_here', action='store_true', default='', help='save near input')
    parser.add_argument('--debug', action='store_true', default='', help='save near input')
    args = parser.parse_args()
    print('check file')

    setattr(args, 'input', os.path.abspath(args.input))
    if args.save_here:
        parent = Path(args.input).parent
        basename = os.path.basename(args.input).split('.')[0]
        output_path = opj(str(parent), f'{basename}_dedup.txt')
        setattr(args, 'output', output_path)
        print('output path')
        print(output_path)
    setattr(args, 'output', os.path.abspath(args.output))
    check_file(args.output)

    print('gather image paths...')
    if os.path.isdir(args.input):
        image_paths = get_image_paths_under_folder(args.input)
        other_infos = []
    elif args.input.endswith('.txt'):
        image_paths, other_infos = ann_loader(args.input)
    else:
        raise NotImplementedError(f'Only support folder and txt. Not support for {args.input}')

    if args.debug:
        print('Debug mode On')
        image_paths = image_paths[:5]
        other_infos = other_infos[:5]

    print('Hashing...')
    pool = ParallelElements(args.n_pools)
    res = pool.run(get_image_info, image_paths)  # [(md5, phash)]
    print('Hashing Done examples:')
    print(res[:5])

    infos = []
    for image_path, other_info, (md5, phash, (h, w)) in zip(image_paths, other_infos, res):
        if args.label != '':
            label = args.label
        elif len(other_info) != '':
            label = other_info
        else:
            label = ''
        info = {
            'path': image_path,
            'label': label,
            'md5': md5,
            'phash': phash,
            'size': (h, w)
        }
        infos.append(info)

    print(len(infos))

    infos = image_path_dedup(infos)
    infos_bad_removed, trace_bad = remove_bad_images(infos)
    infos_md5_deduped, trace_md5 = md5_dedup(infos_bad_removed)
    infos_final, trace_phash = phash_dedup(infos_md5_deduped, args.phash_thresh)
    trace = {'bad': trace_bad, 'md5': trace_md5, 'phash': trace_phash}

    # format output
    print('format output')
    pe = ParallelElements(args.n_pools)
    out_infos = pe.run(format_output, infos_final)

    print('example infos:')
    print(out_infos[0])
    print(f'Before: {len(image_paths)}, After: {len(out_infos)}')

    with open(args.output, 'w') as f:
        f.writelines(out_infos)
    print(args.output)
    print('Done')
