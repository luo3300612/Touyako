from glob import glob
import json
from os.path import join as opj
from collections import defaultdict
import pickle as pkl
from tqdm import tqdm
import random
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch
# import imagesize
import os
import torch.nn as nn
import argparse
from .data_structure import UnionFind


def gather_infos(infos):
    # infos [{[path:, label:, md5:, phash:, size:}]
    res = {key: [] for key in infos[0].keys()}
    n_images = 0
    n_removed = 0
    for info in infos:
        for key in info.keys():
            res[key].append(info[key])
        n_images += 1
    return res


def image_path_dedup(infos):
    # 这一步很重要
    # 如果仅仅是path相同
    # md5去重的时候就会把所有重复的path全部去掉
    # 这样这个path的图片就没有了
    image_path_set = set()
    out_infos = []
    for info in infos:
        image_path = info['path']
        if image_path not in image_path_set:
            image_path_set.add(image_path)
            out_infos.append(info)
    print(f'Remove path dup: {len(infos)}=>{len(out_infos)} (remove {len(out_infos) - len(infos)})')
    return out_infos


def get_bad_images(image_paths, md5s):
    images_to_remove = []
    for image_path, md5 in zip(image_paths, md5s):
        if md5 == -1:  # md5 of bad image is set to 1
            images_to_remove.append(image_path)
    return images_to_remove


def remove_bad_images(infos):
    infos_dict = gather_infos(infos)
    image_paths = infos_dict['path']
    md5s = infos_dict['md5']
    images_to_remove = get_bad_images(image_paths, md5s)
    out_infos = remove_images(infos, images_to_remove, note='Remove bad images')
    return out_infos, images_to_remove


def remove_images(infos, image_paths, note='Remove images'):
    res = []
    image_paths = set(image_paths)
    for info in infos:
        if info['path'] not in image_paths:
            res.append(info)

    n_origin = len(infos)
    n_current = len(res)
    print(f'{note}: {n_origin}=>{n_current} (remove {n_origin - n_current})')
    return res


def md5_dedup(infos):
    infos_dict = gather_infos(infos)
    image_paths = infos_dict['path']
    md5s = infos_dict['md5']

    md5_dict = {}  # { md5: [fpath1, fpath2] }
    duplicated_md5s = set()
    for image_path, md5 in tqdm(zip(image_paths, md5s)):
        if md5_dict.get(md5, None) is None:
            md5_dict[md5] = [image_path]
        else:
            md5_dict[md5].append(image_path)
            duplicated_md5s.add(md5)

    images_to_remove = []
    for md5 in duplicated_md5s:
        images_to_remove += md5_dict[md5][1:]  # 保留第一张

    infos_md5_deduped = remove_images(infos, images_to_remove, note='Remove MD5 dup')

    duplicated_image_sets = []
    for md5 in duplicated_md5s:
        duplicated_image_sets.append(md5_dict[md5])
    return infos_md5_deduped, duplicated_image_sets


def get_sim_pairs(phashs, thresh):
    phash_matrix = np.array([phash for phash in phashs])
    phash_matrix = torch.from_numpy(phash_matrix).float()
    phash_matrix = phash_matrix.flatten(1)  # n * 64
    phash_matrix = phash_matrix.unsqueeze(0)  # 1 * n * 64
    phash_matrix = phash_matrix.cuda()

    normed_phash_matrix = (phash_matrix - 0.5) * 2
    normed_phash_matrix_T = normed_phash_matrix.transpose(1, 2).contiguous()  # 1 * 64 * n
    current_normed_phash_matrix_T = normed_phash_matrix_T  # 1 * 64 * bs

    with torch.no_grad():
        batch_size = 250
        # thresh = args.phash_thresh
        n_batch = int((phash_matrix.shape[1] - 1) / batch_size) + 1
        res = []
        for i in tqdm(range(n_batch), desc='computing distance'):
            batch_start_index = i * batch_size
            batch_data = normed_phash_matrix[:, i * batch_size:(i + 1) * batch_size, :]

            batch_data = batch_data.contiguous()
            current_normed_phash_matrix_T = current_normed_phash_matrix_T.contiguous()

            d = (64 - torch.matmul(batch_data, current_normed_phash_matrix_T)) / 2  # faster than cdist

            xs, ys = torch.where(d[0] < thresh)

            xs = xs.cpu().numpy().tolist()
            ys = ys.cpu().numpy().tolist()

            for x, y in zip(xs, ys):
                res.append(set((x + batch_start_index, y + batch_start_index)))
            current_normed_phash_matrix_T = current_normed_phash_matrix_T[:, :, batch_size:]  # faster as iteration goes
    return res


def get_dup_sets(sim_pairs, n_samples):
    # 根据相似的pari通过并查集得到不相交的相似集合
    uf = UnionFind(n_samples)
    for item in tqdm(sim_pairs, desc='build unionfind'):  # 合并相似的图片到一个集合
        if len(item) == 2:
            item = list(item)
            uf.union(item[0], item[1])

    sets = defaultdict(list)  # 构建相似图片的集合
    for i in tqdm(range(n_samples), desc='find sim image set'):
        sets[uf.find(i)].append(i)

    dup_sets = []  # 找到图片数大于等于2的集合，即重复的图片
    for s in tqdm(sets.values(), desc='find dup set'):
        if len(s) >= 2:
            dup_sets.append(s)
    return dup_sets


def get_images_to_remain(dup_sets, image_paths, image_sizes):
    # 在所有重复集合中，保留大于224(训练尺寸)的最小的图片
    images_to_remain = []
    trace = []  # 用来check去重的结果
    for dup_set in tqdm(dup_sets):
        max_size_prod = 0
        max_size_index = -1

        min_size_prod = 99999 * 99999
        min_size_index = -1

        cur_dup_images = []
        for index in dup_set:
            cur_dup_images.append(image_paths[index])
            h, w = image_sizes[index]  # get image size without open it
            size_prod = h * w

            if h >= 224 and w >= 224 and size_prod < min_size_prod:  # keep smallest image bigger than (224,224)
                min_size_prod = size_prod
                min_size_index = index

            if size_prod > max_size_prod:  # keep largest image
                max_size_prod = size_prod
                remain_index = index

        if min_size_index != -1:
            keep_path = image_paths[min_size_index]
        else:
            keep_path = image_paths[max_size_index]

        images_to_remain.append(keep_path)
        trace.append({'keep': keep_path, 'dup_set': cur_dup_images})
    return images_to_remain, trace


def phash_dedup(infos, thresh, checkable=False):
    infos_dict = gather_infos(infos)
    image_paths = infos_dict['path']
    phashs = infos_dict['phash']

    sim_pairs = get_sim_pairs(phashs, thresh)
    dup_sets = get_dup_sets(sim_pairs, n_samples=len(phashs))

    image_sizes = infos_dict['size']
    images_to_remain, trace = get_images_to_remain(dup_sets, image_paths, image_sizes)

    images_to_remain = set(images_to_remain)
    images_to_delete = []
    for dup_set in tqdm(dup_sets, desc='collect images to delete'):
        for index in dup_set:
            image_path = image_paths[index]
            if image_path not in images_to_remain:
                images_to_delete.append(image_paths[index])

    infos_out = remove_images(infos, images_to_delete, 'Remove phash')
    return infos_out, trace


def check_file(args):
    if os.path.exists(args.output):  # 防止覆盖
        print(f'output file {args.output} exists! overwrite? (y?)')
        c = input()
        if c != 'y':
            exit(0)

    filename = args.output.split('/')[-1]
    folder = args.output[:-len(filename)]

    if not os.path.exists(folder):
        print(f'Folder {folder}, not exists!')
        exit(0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='input image path file')
    parser.add_argument('--output', type=str, help='output info file')
    parser.add_argument('--n-gpus', type=int, help='number of avaliable gpus')
    parser.add_argument('--phash-thresh', type=int, default=3, help='thresh for phash dedup')
    # parser.add_argument('--n-pools', type=int, default=32, help='number of max pools')
    args = parser.parse_args()

    check_file(args)
    print('Loading pkls..')

    infos = pkl.load(open(args.input, 'rb'))
    infos = image_path_dedup(infos)
    infos_bad_removed, trace_bad = remove_bad_images(infos)
    infos_md5_deduped, trace_md5 = md5_dedup(infos_bad_removed)
    infos_final, trace_phash = phash_dedup(infos_md5_deduped, args.phash_thresh)
    trace = {'bad': trace_bad, 'md5': trace_md5, 'phash': trace_phash}

    print('Saving out file...')
    pkl.dump(infos_final, open(args.output, 'wb'))
    output_filename_pre = args.output.split('.pkl')[0]
    print('Saving trace file...')
    pkl.dump(trace, open(output_filename_pre + '_trace.pkl', 'wb'))
    print('Done')
