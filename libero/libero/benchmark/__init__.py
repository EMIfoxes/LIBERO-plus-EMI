import abc
import os
import glob
import random
import torch
import re

from typing import List, NamedTuple, Type
from libero.libero import get_libero_path
from libero.libero.benchmark.libero_suite_task_map import libero_task_map
import libero.libero.envs.bddl_utils as BDDLUtils

BENCHMARK_MAPPING = {}


def register_benchmark(target_class):
    """We design the mapping to be case-INsensitive."""
    BENCHMARK_MAPPING[target_class.__name__.lower()] = target_class


def get_benchmark_dict(help=False):
    if help:
        print("Available benchmarks:")
        for benchmark_name in BENCHMARK_MAPPING.keys():
            print(f"\t{benchmark_name}")
    return BENCHMARK_MAPPING


def get_benchmark(benchmark_name):
    return BENCHMARK_MAPPING[benchmark_name.lower()]


def print_benchmark():
    print(BENCHMARK_MAPPING)


class Task(NamedTuple):
    name: str
    language: str
    problem: str
    problem_folder: str
    bddl_file: str
    init_states_file: str


def grab_language_from_filename(suite_name, x):
    if "_language_" not in x:
        if x[0].isupper():  # LIBERO-100
            if "SCENE10" in x:
                language = " ".join(x[x.find("SCENE") + 8 :].split("_"))
            else:
                language = " ".join(x[x.find("SCENE") + 7 :].split("_"))
        else:
            language = " ".join(x.split("_"))
        en = language.find(".bddl")
        return language[:en]
    else:
        # === 修改开始：动态获取绝对路径 ===
        # 1. 获取当前文件 (benchmark/__init__.py) 的目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. 回退一层到 libero/libero 目录
        libero_root = os.path.dirname(current_dir)
        # 3. 拼接得到正确的 bddl_files 路径
        bddl_files_root = os.path.join(libero_root, "bddl_files")
        # ================================

        if "_view_" in x:
            bddl_file_path = os.path.join(
                bddl_files_root,  # 使用我们动态获取的路径
                suite_name,
                x.split("_view_")[0]+'.bddl',
            )
        else:
            bddl_file_path = os.path.join(
                bddl_files_root,  # 使用我们动态获取的路径
                suite_name,
                x,
            )
            
        problem_info = BDDLUtils.get_problem_info(bddl_file_path)
        return problem_info["language_instruction"]


libero_suites = [
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_10",
    "libero_mix",
]
task_maps = {}
max_len = 0
for libero_suite in libero_suites:
    task_maps[libero_suite] = {}

    for task in libero_task_map[libero_suite]:
        language = grab_language_from_filename(libero_suite, task + ".bddl")
        task_maps[libero_suite][task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=libero_suite,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(language, "\n", f"{task}.bddl", "\n")
        # print("")

suite_order = ["libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90","libero_mix"]
task_num = [2402, 2518, 2591, 2519, 90]
task_order_dict = dict()

for idx in range(5):
    task_orders = [list(range(0,task_num[idx]))]
    for _ in range(19):
        order = list(range(0,task_num[idx]))
        random.shuffle(order)
        task_orders.append(order)
    task_order_dict[suite_order[idx]] = task_orders

if "libero_mix" in task_maps:
    libero_mix_task_count = len(task_maps["libero_mix"])
    print(f"[info] dynamically calculating task orders for libero_mix with {libero_mix_task_count} tasks")
    
    task_orders = [list(range(0, libero_mix_task_count))]
    for _ in range(19):
        order = list(range(0, libero_mix_task_count))
        random.shuffle(order)
        task_orders.append(order)
    task_order_dict["libero_mix"] = task_orders
else:
    # 如果 libero_mix 不在 task_maps 中，使用默认值
    default_task_count = 100
    print(f"[warning] libero_mix not found in task_maps, using default task count {default_task_count}")
    
    task_orders = [list(range(0, default_task_count))]
    for _ in range(19):
        order = list(range(0, default_task_count))
        random.shuffle(order)
        task_orders.append(order)
    task_order_dict["libero_mix"] = task_orders

class Benchmark(abc.ABC):
    """A Benchmark."""

    def __init__(self, task_order_index=0):
        self.task_embs = None
        self.task_order_index = task_order_index

    def _make_benchmark(self):
        tasks = list(task_maps[self.name].values())
        # print(f"[info] using task orders {task_order_dict[self.name][self.task_order_index]}")
        self.tasks = [tasks[i] for i in task_order_dict[self.name][self.task_order_index]]
        self.n_tasks = len(self.tasks)

    def get_num_tasks(self):
        return self.n_tasks

    def get_task_names(self):
        return [task.name for task in self.tasks]

    def get_task_problems(self):
        return [task.problem for task in self.tasks]

    def get_task_bddl_files(self):
        return [task.bddl_file for task in self.tasks]

    def get_task_bddl_file_path(self, i):
        bddl_file_path = os.path.join(
            get_libero_path("bddl_files"),
            self.tasks[i].problem_folder,
            self.tasks[i].bddl_file,
        )
        return bddl_file_path

    def get_task_demonstration(self, i):
        assert (
            0 <= i and i < self.n_tasks
        ), f"[error] task number {i} is outer of range {self.n_tasks}"
        # this path is relative to the datasets folder
        demo_path = f"{self.tasks[i].problem_folder}/{self.tasks[i].name}_demo.hdf5"
        return demo_path

    def get_task(self, i):
        return self.tasks[i]

    def get_task_emb(self, i):
        return self.task_embs[i]

    def get_task_init_states(self, i):
        # === 修改开始：动态获取绝对路径 ===
        # 1. 获取 benchmark 文件夹路径
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # 2. 回退一层到 libero/libero 目录
        libero_lib_root = os.path.dirname(current_dir)
        # 3. 拼接得到 init_files 的绝对路径 (对应报错中的 .../init_files)
        init_files_root = os.path.join(libero_lib_root, "init_files")
        # ================================

        # 初始化变量，防止后面没有匹配到任何条件导致报错
        init_states_path = None

        # 逻辑保持原样，但将 get_libero_path("init_states") 替换为 init_files_root
        if "_language_" in self.tasks[i].init_states_file:
            init_states_path = os.path.join(
                init_files_root,  # 🔥 替换这里
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file.split("_language_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
            )
        else:
            if "_view_" in self.tasks[i].init_states_file:
                init_states_path = os.path.join(
                    init_files_root,  # 🔥 替换这里
                    self.tasks[i].problem_folder,
                    self.tasks[i].init_states_file.split("_view_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
                )
            else:
                # 这里是一系列的 if，处理不同情况
                if "_table_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        init_files_root,  # 🔥 替换这里
                        self.tasks[i].problem_folder,
                        re.sub(r'_table_\d+', '', self.tasks[i].init_states_file),
                    )
                
                if "_tb_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        init_files_root,  # 🔥 替换这里
                        self.tasks[i].problem_folder,
                        re.sub(r'_tb_\d+', '', self.tasks[i].init_states_file),
                    )
                
                if "_light_" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        init_files_root,  # 🔥 替换这里
                        self.tasks[i].problem_folder,
                        self.tasks[i].init_states_file.split("_light_")[0] + "." + self.tasks[i].init_states_file.split(".")[-1],
                    )
                
                if "_add_" in self.tasks[i].init_states_file or "_level" in self.tasks[i].init_states_file:
                    init_states_path = os.path.join(
                        init_files_root,  # 🔥 替换这里
                        "libero_newobj",
                        self.tasks[i].problem_folder,
                        self.tasks[i].init_states_file,
                    )

        # 🔥 兜底逻辑：如果上面所有 if 都没有匹配到（比如普通的文件名），使用默认路径
        # 原代码注释掉了这个 else，建议加上，否则普通任务会报错
        if init_states_path is None:
            init_states_path = os.path.join(
                init_files_root,
                self.tasks[i].problem_folder,
                self.tasks[i].init_states_file,
            )
        
        # print("====init_states_path=====", init_states_path)

        init_states = torch.load(init_states_path,weights_only=False)
        if "_add_" in self.tasks[i].init_states_file or "_level" in self.tasks[i].init_states_file:
            init_states = init_states.reshape(1, -1)
        return init_states

    def set_task_embs(self, task_embs):
        self.task_embs = task_embs


@register_benchmark
class LIBERO_SPATIAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_spatial"
        self._make_benchmark()


@register_benchmark
class LIBERO_OBJECT(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_object"
        self._make_benchmark()


@register_benchmark
class LIBERO_GOAL(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_goal"
        self._make_benchmark()


@register_benchmark
class LIBERO_90(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        assert (
            task_order_index == 0
        ), "[error] currently only support task order for 10-task suites"
        self.name = "libero_90"
        self._make_benchmark()


@register_benchmark
class LIBERO_10(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_10"
        self._make_benchmark()


@register_benchmark
class LIBERO_100(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_100"
        self._make_benchmark()

@register_benchmark
class LIBERO_MIX(Benchmark):
    def __init__(self, task_order_index=0):
        super().__init__(task_order_index=task_order_index)
        self.name = "libero_mix"
        self._make_benchmark()
