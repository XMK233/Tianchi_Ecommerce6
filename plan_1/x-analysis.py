import pandas as pd
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from config import Config

def analyze_data():
    print("="*50)
    print("正在加载数据...")
    print("="*50)
    
    # Load Data
    train_reviews = pd.read_csv(Config.TRAIN_REVIEWS)
    train_labels = pd.read_csv(Config.TRAIN_LABELS)
    test_reviews = pd.read_csv(Config.TEST_REVIEWS)
    
    # Merge train reviews and labels
    train_full = pd.merge(train_reviews, train_labels, on='id', how='left')
    
    print(f"训练集评论数: {len(train_reviews)}")
    print(f"训练集标注条目数: {len(train_labels)}")
    print(f"测试集评论数: {len(test_reviews)}")
    print("-" * 30)

    # --- 1. 文本长度分析 ---
    print("\n[1. 文本长度分析]")
    train_reviews['len'] = train_reviews['Reviews'].astype(str).apply(len)
    test_reviews['len'] = test_reviews['Reviews'].astype(str).apply(len)
    
    print(f"训练集平均长度: {train_reviews['len'].mean():.2f} (Max: {train_reviews['len'].max()}, Min: {train_reviews['len'].min()})")
    print(f"测试集平均长度: {test_reviews['len'].mean():.2f} (Max: {test_reviews['len'].max()}, Min: {test_reviews['len'].min()})")
    
    # --- 2. 类别分布 (Category) ---
    print("\n[2. 评价类别分布 (Category)]")
    cat_counts = train_labels['Categories'].value_counts()
    print(cat_counts)
    
    # Top 3 categories
    top_cats = cat_counts.head(3).index.tolist()
    print(f"-> 用户最关注的前三大维度: {', '.join(top_cats)}")

    # --- 3. 情感倾向分布 (Polarity) ---
    print("\n[3. 情感倾向分布 (Polarity)]")
    pol_counts = train_labels['Polarities'].value_counts()
    print(pol_counts)
    
    pos_ratio = pol_counts.get('正面', 0) / len(train_labels)
    neg_ratio = pol_counts.get('负面', 0) / len(train_labels)
    print(f"-> 正面评价占比: {pos_ratio:.2%}")
    print(f"-> 负面评价占比: {neg_ratio:.2%}")

    # --- 4. 类别 x 情感 交叉分析 ---
    print("\n[4. 各类别的‘好评/差评’分析]")
    # Cross tabulation
    ct = pd.crosstab(train_labels['Categories'], train_labels['Polarities'])
    # Calculate negative rate for each category
    if '负面' in ct.columns:
        ct['Negative_Rate'] = ct['负面'] / ct.sum(axis=1)
        ct_sorted = ct.sort_values('Negative_Rate', ascending=False)
        print("按‘负面率’排名的类别 (Top 5 易被吐槽的点):")
        print(ct_sorted[['正面', '负面', 'Negative_Rate']].head(5))
    else:
        print("数据中未发现‘负面’标签")

    # --- 5. Aspect 和 Opinion 分析 ---
    print("\n[5. 核心词汇分析]")
    
    # Filter out empty/placeholder
    valid_aspects = train_labels[train_labels['AspectTerms'] != '_']['AspectTerms']
    valid_opinions = train_labels[train_labels['OpinionTerms'] != '_']['OpinionTerms']
    
    print(f"隐式Aspect占比: {(train_labels['AspectTerms'] == '_').mean():.2%}")
    
    print("\nTop 10 显式评价对象 (Aspects):")
    print(valid_aspects.value_counts().head(10))
    
    print("\nTop 10 评价词 (Opinions):")
    print(valid_opinions.value_counts().head(10))

    # --- 6. 热门组合 (Aspect + Opinion) ---
    print("\n[6. 热门‘对象-观点’组合]")
    train_labels['Pair'] = train_labels['AspectTerms'] + " - " + train_labels['OpinionTerms']
    print(train_labels['Pair'].value_counts().head(10))
    
    print("="*50)
    print("分析完成")
    print("="*50)

if __name__ == "__main__":
    analyze_data()


'''

==================================================
正在加载数据...
==================================================
训练集评论数: 3229
训练集标注条目数: 6633
测试集评论数: 2237
------------------------------

[1. 文本长度分析]
训练集平均长度: 21.70 (Max: 69, Min: 1)
测试集平均长度: 21.30 (Max: 72, Min: 2)

[2. 评价类别分布 (Category)]
Categories
整体      2822
使用体验    1042
功效       726
价格       696
物流       517
气味       225
包装       195
真伪       161
服务        86
其他        65
成分        61
尺寸        24
新鲜度       13
Name: count, dtype: int64
-> 用户最关注的前三大维度: 整体, 使用体验, 功效

[3. 情感倾向分布 (Polarity)]
Polarities
正面    5925
负面     556
中性     152
Name: count, dtype: int64
-> 正面评价占比: 89.33%
-> 负面评价占比: 8.38%

[4. 各类别的‘好评/差评’分析]
按‘负面率’排名的类别 (Top 5 易被吐槽的点):
Polarities   正面  负面  Negative_Rate
Categories                        
新鲜度           1  11       0.846154
尺寸            7  15       0.625000
包装          139  51       0.261538
其他           50  15       0.230769
气味          173  44       0.195556

[5. 核心词汇分析]
隐式Aspect占比: 71.39%

Top 10 显式评价对象 (Aspects):
AspectTerms
物流      196
价格      140
包装      134
味道      128
活动       90
补水效果     75
快递       69
赠品       52
保湿效果     39
速度       37
Name: count, dtype: int64

Top 10 评价词 (Opinions):
OpinionTerms
不错     444
很好     329
很好用    206
还不错    163
还可以    147
好用     135
挺好的    135
很快     133
很喜欢    115
快       98
Name: count, dtype: int64

[6. 热门‘对象-观点’组合]
Pair
_ - 不错      353
_ - 很好用     203
_ - 很好      187
_ - 还不错     149
_ - 好用      134
_ - 还可以     133
_ - 挺好的     120
_ - 很喜欢     109
_ - 好评       90
_ - 挺好用的     85
Name: count, dtype: int64
==================================================
分析完成
==================================================


通过对训练集数据的深入分析，我们得到以下 核心业务认知 ：

### 1. 评论特点：短文本、隐式对象多
- 平均长度约 21 字 ：典型的电商短评，言简意赅。
- 隐式 Aspect 占比高达 71.39% ：绝大多数评论没有明确指出评价对象（例如直接说“很好用”，而不是“ 洗面奶 很好用”）。这说明模型在预测时，需要具备很强的 上下文推理能力 ，或者直接预测 None / _ 作为 Aspect。
### 2. 用户关注点（Categories）
- Top 3 关注维度 ：
  1. 整体 (2822条)：大部分用户习惯给出一个笼统的评价。
  2. 使用体验 (1042条)：产品好不好用是第二大核心。
  3. 功效 (726条)：用户很看重产品实际产生的效果。
  - 启示 ：模型需要重点区分“整体评价”和具体的“体验/功效”描述，这三者占据了绝大多数数据，是提分的关键。
### 3. 情感倾向：好评如潮，差评集中
- 正面评价占比 89.33% ：数据严重不平衡，绝大多数都是好评。
- 负面评价占比 8.38% ：差评虽少，但非常关键。
- “重灾区”（高负面率类别） ：
  1. 新鲜度 (负面率 84.6%)：几乎只要提到新鲜度，就是在骂（可能涉及保质期、生产日期）。
  2. 尺寸 (负面率 62.5%)：提到尺寸通常是因为不合适或太小。
  3. 包装 (负面率 26.1%)：包装破损或简陋是常见的吐槽点。
  - 业务价值 ：虽然“新鲜度”和“尺寸”总样本量少，但 负面风险极高 。如果是做舆情监控，这几个维度出现时要自动触发红色预警。
### 4. 核心词汇 (Aspects & Opinions)
- 显式评价对象 (Aspects) ：
  - 物流 (196), 价格 (140), 包装 (134), 味道 (128) 是最常被显式提及的名词。
- 高频评价词 (Opinions) ：
  - 不错 , 很好 , 很好用 , 还不错 , 还可以 占据统治地位。
- 热门组合 ：
  - _ - 不错 是出现频率最高的组合（353次）。这也验证了前面的隐式 Aspect 占比高的问题，用户习惯直接用形容词表达观点。
### 总结与建议
1. 模型优化方向 ：由于 71% 的 Aspect 是隐式的，我们在做提取任务时，必须强化对 Implicit Aspect (即 _ 或 None ) 的处理能力。如果模型总是强行找一个名词当 Aspect，准确率会很低。
2. 数据增强 ：负面样本（尤其是 新鲜度 、 尺寸 等类别）较少，可以考虑对这些类别的负面样本进行过采样或专门的数据增强，以提高模型对差评的召回率。
3. 业务应用 ：商家应重点关注 物流 和 包装 ，虽然它们不是最核心的关注点，但却是最容易引发差评的显性因素；而 功效 和 使用体验 则是获取好评的关键。'''