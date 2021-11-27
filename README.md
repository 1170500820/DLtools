## DLtools

任务相关代码与常用工具解耦合，方便多人协同工作，提高开发效率。



详细设计见[notion-DLtools文档](https://gierere.notion.site/DLtools-b6628731906a448085248f0c3dd78cea)



### 目录简介

- work

  存放任务相关的代码。尽量最小化实现一个新的任务所需的代码，把可复用的代码放在其他目录下。

- utils

  常用的工具函数，比如序列处理、batchify

- models

  常用模型组件

- evaluate

  常用的评价函数与评价组件

- train

  实现几个可通用的训练loop

- analysis

  数据分析模块。

