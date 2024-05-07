def load_anns(ann_paths):
    outputs = []
    for ann_path in ann_paths:
        with open(ann_path) as f:
            outputs += f.readlines()
    print(f'load {len(outputs)} anns')
    return outputs