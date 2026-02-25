toolName: view_files
            
status: success
          
            
filePath: /mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_3-い-基于2-样本增强/config.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_3-い-基于2-样本增强/utils_loc.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_3-い-基于2-样本增强/model.py
          

        
            
toolName: view_files
            
status: success
          
            
filePath: /mnt/d/forCoding_code/Tianchi_Ecommerce6/plan_3-い-基于2-样本增强/train.py
          
针对 `plan_3` (基于 RoBERTa + 样本增强) 的代码，为了进一步提升模型在 ABSA 任务上的表现，我为您整理了三种在 NLP 竞赛和实践中非常有效且通用的代码增强方案：**对抗训练 (FGM)**、**指数移动平均 (EMA)** 和 **Multi-Sample Dropout**。

以下是具体的方案介绍和代码样例。

### 方案一：对抗训练 (FGM)

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

### 方案二：指数移动平均 (EMA)

**原理**：维护模型参数的滑动平均值。在测试/预测时，使用滑动平均后的参数而不是最后一次迭代的参数。这通常能让模型收敛到更平坦（更稳定）的极小值区域。
**适用场景**：所有模型训练，能有效提升最后几个点的稳定性。

**代码样例**：

定义 `EMA` 类：

```python
class EMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
```

**集成到 `train.py`**：

```python
# 初始化
ema = EMA(model, decay=0.999)
ema.register()

for batch in loader:
    # ... (训练代码) ...
    optimizer.step()
    ema.update() # 更新 EMA 参数
    # ...

# --- 验证/预测阶段 ---
ema.apply_shadow() # 应用 EMA 参数到模型
# ... (执行验证或保存模型) ...
# torch.save(model.state_dict(), ...) # 保存的是 EMA 后的参数
ema.restore() # 恢复原始参数继续训练（如果需要）
```

---

### 方案三：Multi-Sample Dropout

**原理**：在 BERT 输出层后使用多个不同的 Dropout 层，然后将结果取平均。这可以加速收敛并提供类似集成的正则化效果。
**适用场景**：修改 `model.py` 中的模型结构。

**代码样例**：

修改 `model.py` 中的 `ExtractionModel` (分类模型同理)：

```python
class ExtractionModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        
        # 移除单个 dropout
        # self.dropout = nn.Dropout(0.1)
        
        # 定义多个 dropout
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.classifier = nn.Linear(config.hidden_size, len(Config.TAGS))
        
    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        
        # Multi-Sample Dropout Logic
        logits = None
        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                out = dropout(sequence_output)
                logits = self.classifier(out)
            else:
                out = dropout(sequence_output)
                logits += self.classifier(out)
        
        logits = logits / len(self.dropouts) # 取平均
        
        # ... (后续计算 Loss 逻辑不变) ...
```

### 建议

如果您希望快速尝试，建议优先加入 **FGM (对抗训练)**。它对代码结构的改动主要在训练循环中，且通常能带来 0.5% - 1% 的 F1 提升。

如果您需要，我可以为您**创建一个包含 FGM 和 EMA 的工具文件**，并帮您修改 `train.py` 来集成这些功能。您看如何？