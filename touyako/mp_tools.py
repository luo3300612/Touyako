import math
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


class ParallelLists:
    # 并行List
    # 将List划分为n个子List，让func进行并行处理
    def __init__(self, processes):
        self.processes = processes
        self.pool = Pool(processes=processes)

    def prepare_chunks(self, args: list):
        # 通常需要重新写
        # 这里给一个仅以列表为参数的func的参数划分方法
        # 划分成self.processes个子列表，分给每个进程
        n_elements = math.ceil(len(args) / self.processes)
        res = []
        for i in range(self.processes):
            res.append(args[i * n_elements:(i + 1) * n_elements])
        return res

    def run(self, func, **kwargs):
        param_chunks = self.prepare_chunks(**kwargs)
        outputs = []
        for i, param in enumerate(param_chunks):
            outputs.append(self.pool.apply_async(func, (*param,)))
        self.pool.close()
        self.pool.join()

        return self.join_results(outputs)

    def join_results(self, outputs):
        # 根据func的输出内容不同需要重写
        res = []
        for output in outputs:
            res += output.get()
        return res


class ParallelElements:
    # 多进程
    # 并行Elements
    # 元素级的并行
    # 支持tqdm
    def __init__(self, processes):
        self.processes = processes
        self.pool = Pool(processes=processes)

    def run(self, func, args: list):
        pbar = tqdm(total=len(args))
        update = lambda *args: pbar.update()

        outputs = []
        for i, arg in enumerate(args):
            outputs.append(self.pool.apply_async(func, (arg,), callback=update))
        self.pool.close()
        self.pool.join()
        res = []
        for output in outputs:
            res.append(output.get())
        return res
