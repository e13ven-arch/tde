# 01 · 权重文件是 fp16 存的，模型一步都没学

**结论先说**：DeBERTa-v3-base 训了 2,188 步，损失从第 25 步起就平在 1.1，准确率 40%，等于随机。原因不是模型、不是数据、不是学习率，是权重文件本身以 fp16 存储，transformers 5 按文件精度加载，整个骨干以 fp16 参数做 AdamW 更新，更新量比 fp16 的最小分辨率还小，全部被吃掉。

**怎么发现的**

同一份代码、同一份数据，ModernBERT 正常收敛，DeBERTa 纹丝不动。先怀疑分词器（DeBERTa 的 spm.model 在 transformers 5 里确实加载失败，修了一轮），再怀疑 bf16 autocast，关掉 autocast 直接报错：

```
RuntimeError: expected scalar type Half but found Float
```

这句报错是线索：LayerNorm 收到了 Half 权重。打印参数 dtype：

```
Counter({'torch.float16': 198})
```

198 个参数张量全是 fp16。config.json 里没有 dtype 字段，是 pytorch_model.bin 本身存的 fp16。旧版 transformers 默认转成 fp32 加载，v5 改成保留文件精度，所以老代码换新库就中招。

**为什么不报错**

fp16 权重 + autocast bf16 前向不报错，loss 有值，梯度有值，优化器也在跑。只是 2e-5 的学习率乘上梯度，加到 fp16 的权重上，绝大多数更新小于 fp16 在该量级的间距，舍入后等于没加。看曲线像"学不动"，实际是"没在学"。

**怎么避免**

加载编码器时显式给 dtype：

```python
AutoModel.from_pretrained(name, dtype=torch.float32)
```

fp32 主权重，bf16 autocast 算前向，这是混合精度的标准做法。另外训练开始后看两行日志就够：参数 dtype 是不是 fp32，前 100 步 loss 有没有动。这两行本该在第一次跑的时候就看。

修好之后同一配置：准确率 0.882，NLL 0.301，比 ModernBERT-base 高 1.9 个点。

---
仓库：github.com/e13ven-arch/tde · 实验记录 docs/RESULTS_exp005.md（Exp 018）· 修复提交 "Load encoder backbones as fp32 master weights"
