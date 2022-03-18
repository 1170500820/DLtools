import sys
sys.path.append('.')
from typing import List, Dict, TypedDict, Union, Any, Set, Iterable, Callable, Literal, Tuple, Sequence, Hashable
from collections import namedtuple

Args = namedtuple('args', 'name_or_flags dest type help default')

"""
一些基础的方便使用的类型定义
"""
# int
IntList = List[int]

# str
StrList = List[str]
StrSet = Set[str]

# float
FloatList = List[float]
ProbList = FloatList  # but 0 <= elem <= 1


"""
process相关
"""

# ProcUnit是处理数据的最小单元
ProcUnit = Callable[[dict], dict]


# 自定模块中的参数格式
class MyArgs(TypedDict):
    name: str
    dest: str
    type: type
    help: str
    default: Any

# 事件抽取相关
Span = Tuple[int, int]
SpanList = List[Span]
SpanL = List[int]  # assert length==2
SpanLList = List[SpanL]  # 本来不希望有SpanL这个东西，因为list比tuple难处理一些，没法hash。但是源数据格式用的是list，
SpanSet = Set[Span]
