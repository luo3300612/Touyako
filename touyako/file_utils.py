from tqdm import tqdm
from collections import defaultdict
from os.path import join as opj
import os

valid_image_exts = set(['jpeg', 'png', 'jpg', 'bmp', 'JPG', 'JPEG', 'PNG'])


def txt_loader(fpath):
    """
    :param fpath: txt file path
    :return: txt lines
    """
    outputs = []
    with open(fpath) as f:
        for line in f:
            outputs.append(line.strip())
    return outputs


def ann_loader(fpath):
    image_paths = []
    other_infos = []
    with open(fpath) as f:
        for line in f:
            image_path, *others = line.strip().split('\t')
            image_paths.append(image_path)
            others = '\t'.join(others)
            other_infos.append(others)
    assert len(other_infos) == 0 or len(other_infos) == len(
        image_paths), f"Error in {fpath}, where len(image_paths)={len(image_paths)}, len(other_infos)={len(other_infos)}"
    return image_paths, other_infos


def get_image_paths_under_folder(folder_path, verbose=True):
    """
    :param folder_path:
    :param verbose: whether to print loading information
    :return: image paths
    """
    image_paths = []
    c = defaultdict(int)
    for (dirpath, dirnames, filenames) in tqdm(os.walk(folder_path)):
        for filename in filenames:
            ext = filename.split('.')[-1]
            c[ext] += 1
            if ext in valid_image_exts:
                image_paths.append(opj(dirpath, filename))
    tuple_c = [(k, v) for k, v in c.items()]
    sorted_tuple_c = sorted(tuple_c, key=lambda x: x[1])

    if verbose:
        print('N total paths:', sum(c.values()))
        print('N image valid paths:', len(image_paths))
        print('file classes:')
        print(sorted_tuple_c)
    return image_paths
