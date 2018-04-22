项目概述
---

你将在 https://www.openstreetmap.org 中的世界里选择任意区域，然后使用数据加工技术（比如针对有效性、准确率、完整性、一致性和均匀性评估数据质量），为该世界中你关注的那部分区域清理 OpenStreetMap 数据。最后，你可以选择 MongoDB 或 SQL 完成项目。

我将学习什么？
完成项目后，你将可以：

针对有效性、准确率、完整性、一致性和均匀性来评估数据的质量。
解析并且从 .json、.xml、.csv、.html 等常用文件格式中收集数据。
处理来自大量文件和大型文件并且能够由电子表格程序进行清理的数据。
学习如何使用 MongoDB 或 SQL 存储、查询和聚合数据。

为何要进行这个项目？
从数据库或者各种文件格式中检索数据的难点是什么？你从文件和数据库中获取数据、清理数据和合并数据，然后将其注入到先进的深度学习算法中……对吗？

但实际上，经常和数据打交道的人都知道，从不同数据源获取的数据是无法放在一起使用的，这会造成极大的问题。

此项目将教你如何处理最复杂的数据整理情形。

为何这对我的职业发展很重要？
正如《纽约时报》的这篇文章所指出的，为了后期能够更容易探索和分析数据，人们需要在前期手工收集和清理数据，这部分费时费力的工作被数据科学界称为“数据整理”或者“数据加工”。

尽管不像构建酷炫的机器学习模型那样吸引人，但是根据许多职业数据分析师和数据科学家的说法，数据整理这项任务占用了数据科学家 50%-80% 的时间。

准备项目
---

第一步：完成 数据整理小节课程。

第二步：完成 MongoDB 或 SQL 小节课程。

对于 MongoDB，完成用 MongoDB 进行数据分析小节课程，为此项目做准备。

对于 SQL，完成用 SQL 进行数据分析小节课程，为此项目做准备。

项目详情
对于这个项目，你可以选择使用 SQL 或 MongoDB 来完成。要了解这两种数据库之间的区别，请见这个小节。下面也有针对两种数据库分别的指南。

以下是你要做的事情：

第一步 - 完成编程练习
确保所有“案例分析：OpenStreetMap 数据”中的编程练习都已正确解决（MongoDB 或 SQL，取决于你的选择）。

第二步 - 查看评估准则和示例项目
我们会使用这个项目评估准则审阅你的项目。你需要满足准则中所有的要求。下面是两份报告示例，供你参考

SQL 示例项目
MongoDB 示例项目

第三步 - 选择你的地图区域
从 https://www.openstreetmap.org 中的世界里选择任意区域，然后下载一个 XML OSM 数据集。数据集大小应至少为 50MB（未压缩）。我们建议你使用以下一种方式下载数据集：

从 Map Zen 下载预先选定的都会区域。

使用 Overpass API 下载一个自定义方形区域。语法解释请访问 wiki。通常，你会希望使用以下查询：(node(minimum_latitude, minimum_longitude, maximum_latitude, maximum_longitude);<;);out meta;

例如： (node(51.249,7.148,51.251,7.152);<;);out meta;

要包括 meta 选项，这样元素就可以包含时间戳和用户信息了。你可以使用 Open Street Map 导出工具来查找边界框的坐标。注意：你实际上无法使用导出工具下载数据，因为此项目要求的区域太大。

第四步 - 处理你的数据集
我们建议你从你选择的课程（MongoDB 或 SQL）中的习题集开始，并修改其中代码，以适应你的数据集。当你拆解数据时，记录过程中遇到的问题和有关数据集的问题。你将需要这些信息来撰写项目报告。

提示：你可以从所选区域中的一个小样本开始项目，让你可以更容易地不断改进你的研究。你可以参考下方的代码示例，了解怎样做。

SQL
彻底审查和清理你的数据集，将数据集从 XML 格式转换为 CSV 格式。然后使用这个方法，或自己选择一个方法，将清理后的 .csv 文档导入到 SQL 数据库中。

MongoDB
彻底审查和清理你的数据集，将数据集从 XML 格式转换为 JSON 格式，然后将清理后的 .json文档导入到 MongoDB 数据库中。

第五步 - 探索数据库
搭建好本地数据库后，你要通过查询探索你的数据。请务必在报告中记录下你的查询和查询结果。参考项目评估准则，了解详细查询要求。

第六步 - 记录你的工作
创建一个文档（pdf 或 html 格式），能够回答项目评估准则中的问题。

在地图中遇到的问题
数据概述
关于数据集的其他想法

如果合适，尝试在你的报告中包含代码段和问题标签（请参见 MongoDB 示例项目或 SQL 示例项目）以及可视化。

使用下方的代码，从原始 OSM 区域中获取系统的元素样本。试着改变 k 的值，获得不同大小的样本文件。在开始处理整个数据集前，你可以先使用较大的 k ，然后使用较小的 k 。


``` python
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET  # Use cElementTree or lxml if too slow

OSM_FILE = "some_osm.osm"  # Replace this with your osm file
SAMPLE_FILE = "sample.osm"

k = 10 # Parameter: take every k-th top level element

def get_element(osm_file, tags=('node', 'way', 'relation')):
    """Yield element if it is the right type of tag

    Reference:
    http://stackoverflow.com/questions/3095434/inserting-newlines-in-xml-file-generated-via-xml-etree-elementtree-in-python
    """
    context = iter(ET.iterparse(osm_file, events=('start', 'end')))
    _, root = next(context)
    for event, elem in context:
        if event == 'end' and elem.tag in tags:
            yield elem
            root.clear()


with open(SAMPLE_FILE, 'wb') as output:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write('<osm>\n  ')

    # Write every kth top level element
    for i, element in enumerate(get_element(OSM_FILE)):
        if i % k == 0:
            output.write(ET.tostring(element, encoding='utf-8'))

    output.write('</osm>')
```
