---
layout:     post
title:     	metapath2vec
subtitle:   一种异构图 embedding 算法
date:       2018-10-02
author:     Shaoguang Cheng
header-img: img/metapath2vec-post-bg-graph-embedding.png
catalog: true
tags:
    - graph embedding
---


# metapath2vec


本文是对论文 metapath2vec: scalable representation learning for heterogeneous networks 的总结。

## 论文要点
文章对异构网络和异构网络表示学习给出了形式化定义，提出了 metapath2vec 和 metapath2vec++ 两种算法对异构图网络进行表示学习，经大量实验，作者发现这两种算法不仅在各种异构图挖掘任务（比如 node classification, clustering and similarity search）中表现优秀，而且能够发现不同类型网络节点之间的结构和语义相关性。

## 问题定义

### 异构图网络定义
![异构图定义](../img/metapath2vec-h-graph-def.png)

### 异构图表示学习定义
![异构图学习定义](../img/metapath2vec-h-graph-rep-learning-def.png)

## metapath2vec
metapath2vec 在整体的流程上和 Deep Walk 以及 node2vec 是一致的，都是先通过 random walk 的方式得到每一个节点的上下文节点，然后利用 skip gram 的方式进行节点表示的学习。在 metapath2vec 中，使用了一种 biased random walk。

### metapath based random walk
最直接的做法是直接采用一般的随机游走策略生成上下文序列，但这种效果一般不会很好，主要原因是:  
	1. 多种类型节点的 word-context 问题  
	2. 异构网络上随机游走生成的路径，会偏向最多可见类型的节点

> random walk中，最重要的就是如何定义节点的跳转概率P(u/v)。即根据当前节点 v , 定义出下一次跳转到节点 u 的概率。

文章中提出了使用基于 metapath 的随机游走，metapath 定义了一种由图中节点类型组成的路径模式。例如在论文引用网络中，有作者(A)、论文(P)、机构(O)三种节点，metapath "APA"定义了一种 作者-论文-论文 的路径模式。

给定一个异构网络 G = (V, E, T) 和 metapath, 随机游走的方式定义为：
![metapath-based-random-walk](../img/metapath2vec-metapath-based-rw.png)

基于 metapath 的随机游走策略能确保不同类型的节点的语义相关性体现在生成的上下文路径中。metapath2vec 算法就是将上述生成的上下文序列输入到标准的 skip gram 中，学习各节点的低维稠密向量表示。

## metapath2vec++
metapath2vec 在构建上下文序列的时候考虑了节点的不同类型，能够确保生成的路径符合定义的metapath schema，但在 skip gram 的 softmax 中未考虑节点的类型信息，是在全部的节点中进行负采样的。

metapath2vec++要求根据当前节点对上下文进行预测，进行负采样的时候，不在全部节点中进行采样，而是要在与目标节点同种类型的节点中进行采样。这种做法其实相当于在负采样的时候，进行了 hard example 的筛选，使得学习出来的模型更具有区分能力。

## 几个评价指标
### 分类评价指标
	macro-F1: 对每个类别分别计算 F1 值，所有类别 F1 值的平均即为 macro-F1
	micro-F1: 计算好每个类别的混淆矩阵后，讲混淆矩阵对应元素相加，然后在这个混淆矩阵上计算 F1 值即为micro-F1
	
### 聚类评价指标
	NMI(Normalized Mutual Information) 

## 总结
基于 random walk 的 graph embedding 方法的关键在于如何寻找一种比较好的随机游走方法，deep walk 采用最基本的随机游走， node2vec 采用 DFS 和 BFS (biased random walk)相结合的游走方式，而本文采用基于 metapath 的随机游走方式，来对异构图进行表达学习。




	
	
