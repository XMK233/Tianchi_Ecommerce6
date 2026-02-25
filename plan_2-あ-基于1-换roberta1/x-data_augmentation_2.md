针对 `plan_3` (基于 RoBERTa + 样本增强) 的代码，为了进一步提升模型在 ABSA 任务上的表现，我为您整理了以下代码增强方案。

其中 **对抗训练 (FGM)** 是一种模型层面的正则化增强；而为了增加数据的多样性，这里补充了三种真正的 **数据增强 (Data Augmentation)** 方案：**回译 (Back-Translation)**、**EDA (Easy Data Augmentation)** 和 **Token Masking**。

---

### 方案一：对抗训练 (FGM) [模型增强]

**原理**：在 Embedding 层添加微小的扰动（噪声），强制模型对输入的微小变化保持鲁棒，从而提高泛化能力。
**适用场景**：几乎所有 NLP 任务，尤其是数据量较少或容易过拟合的场景。

**代码样例**：

首先定义 `FGM` 类（建议放在 `utils_loc.py` 或新建 `utils_enhancement.py`）：

```python
import torch

class FGM:
    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1.0, emb_name='word_embeddings'):
        # 在 Embedding 层添加扰动
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='word_embeddings'):
        # 恢复原始 Embedding
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                if name in self.backup:
                    param.data = self.backup[name]
        self.backup = {}
```

**集成到 `train.py` 的训练循环中**：

```python
# 初始化
fgm = FGM(model)

for batch in loader:
    # ... (正常的前向传播和反向传播) ...
    loss.backward() # 第一次反向传播，计算正常梯度

    # --- 对抗训练开始 ---
    fgm.attack() # 1. 在 embedding 上添加对抗扰动
    
    # 2. 再次前向传播和计算损失 (使用扰动后的输入)
    # 注意：这里需要再次调用 model，传入相同的 batch 数据
    loss_adv, _ = model(input_ids, attention_mask, token_type_ids, labels=labels)
    loss_adv.backward() # 3. 反向传播，累加对抗梯度
    
    fgm.restore() # 4. 恢复原始 embedding 参数
    # --- 对抗训练结束 ---

    optimizer.step()
    model.zero_grad()
    scheduler.step()
```

---

### 方案二：回译 (Back-Translation) [数据增强]

**原理**：将中文文本翻译成其他语言（如英语），再翻译回中文。这可以生成语义相同但表达方式不同的新样本。
**适用场景**：
*   **分类任务 (Classification)**：非常适合。因为分类标签（类别/极性）是针对整个句子的（或者针对 Aspect-Opinion 对的），文本的微小变化通常不改变情感。
*   **抽取任务 (Extraction)**：**慎用**。回译会改变原始文本的字词位置和内容，导致原有的 `BIO` 标签索引失效。如果用于抽取任务，需要非常复杂的标签映射算法（Project Labels），或者只对非实体部分进行操作（很难控制）。

**建议**：仅在 **ClassificationDataset** 的构建阶段使用。对原始 Review 进行回译，保持 Aspect/Opinion 词不变（如果它们是句子的一部分，这很难），或者更简单地：**仅对训练集中的短文本进行增强，或者作为一种去噪手段。**

对于本任务的分类阶段（输入是 `[CLS] Review [SEP] Aspect [SEP] Opinion`），可以对 `Review` 部分进行回译增强。

**实现思路**：
使用 `googletrans` 或 `百度翻译API` 预处理数据，生成 `Train_reviews_augmented.csv`，然后在 `train.py` 中混合加载。

```python
# 伪代码
from googletrans import Translator
translator = Translator()

def back_translate(text):
    # zh -> en -> zh
    en = translator.translate(text, src='zh-cn', dest='en').text
    zh = translator.translate(en, src='en', dest='zh-cn').text
    return zh
```

---

### 方案三：EDA (Easy Data Augmentation) [数据增强]

**原理**：使用简单的规则对文本进行操作：
1.  **同义词替换 (SR)**：随机选取非停用词，替换为其同义词。
2.  **随机插入 (RI)**：随机选取非停用词的同义词插入句子中。
3.  **随机交换 (RS)**：随机交换句子中两个词的位置。
4.  **随机删除 (RD)**：以一定概率随机删除句子中的词。

**适用场景**：
*   **分类任务**：完全适用。
*   **抽取任务**：
    *   **随机删除 (RD)**：相对安全，相当于 Dropout。
    *   **同义词替换 (SR)**：如果避开实体词（Aspect/Opinion），是可行的。需要使用分词工具（如 `jieba`）并结合实体位置信息，只替换非实体区域的词。

**代码样例 (针对分类任务的简单 SR)**：

```python
import jieba
import synoyms # 需要安装 synonyms 库: pip install synonyms

def synonym_replacement(text, aspect, opinion, n=1):
    words = list(jieba.cut(text))
    new_words = words.copy()
    
    # 找出 aspect 和 opinion 在分词后的位置，避免替换
    # 这里简化处理：只要词不是 aspect 或 opinion 就尝试替换
    
    replaced_count = 0
    for i, word in enumerate(words):
        if replaced_count >= n: break
        
        if word == aspect or word == opinion: continue
        
        # 获取同义词
        nearby = synonyms.nearby(word)
        if len(nearby[0]) > 0:
            # 选最相似的词
            syn = nearby[0][1] # 0是词本身，1是第一个同义词
            new_words[i] = syn
            replaced_count += 1
            
    return "".join(new_words)
```

---

### 方案四：Token Masking (随机遮盖) [数据增强]

**原理**：在训练过程中，随机将输入序列中的一部分 Token 替换为 `[MASK]` 或 `[UNK]`，或者直接设为 Padding。这迫使模型不依赖特定的某个词，而是利用上下文信息，类似于 BERT 的预训练任务 MLM。

**适用场景**：**抽取任务** 和 **分类任务** 都非常适用且安全，不需要改变 Label。

**代码样例 (集成到 Dataset 的 `__getitem__` 或 Collate_fn 中)**：

```python
class ExtractionDataset(Dataset):
    # ...
    def __getitem__(self, idx):
        # ... 获取 input_ids ...
        
        # Data Augmentation: Random Masking
        # 概率 15% mask 掉，但要避开 [CLS], [SEP]
        if self.is_train: # 只在训练时开启
            seq_len = input_ids.size(0)
            mask_prob = 0.15
            rand = torch.rand(seq_len)
            
            # 创建 mask: 概率小于 mask_prob 且不是特殊 token (假设 101=[CLS], 102=[SEP], 0=[PAD])
            mask_indices = (rand < mask_prob) & (input_ids != 101) & (input_ids != 102) & (input_ids != 0)
            
            # 替换为 [MASK] (BERT mask token id is usually 103)
            # 或者替换为 [UNK] (100)
            input_ids[mask_indices] = 103 
            
        # ...
```

---

### 总结与推荐

| 方案 | 类型 | 抽取任务可行性 | 分类任务可行性 | 推荐指数 | 备注 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **FGM** | 模型增强 | ✅ 高 | ✅ 高 | ⭐⭐⭐⭐⭐ | 必须上，稳赚不赔 |
| **Token Masking** | 数据增强 | ✅ 高 | ✅ 高 | ⭐⭐⭐⭐ | 实现简单，效果稳定 |
| **EDA (同义词)** | 数据增强 | ⚠️ 需避开实体 | ✅ 高 | ⭐⭐⭐ | 需外部词典，实现稍繁 |
| **回译** | 数据增强 | ❌ 难 (坐标偏移) | ✅ 高 | ⭐⭐⭐ | 耗时，适合离线预处理 |

建议您在当前的 FGM 基础上，优先尝试 **Token Masking**，因为它不需要额外的外部资源（翻译API、同义词库），且不会破坏抽取任务的标签对齐。
