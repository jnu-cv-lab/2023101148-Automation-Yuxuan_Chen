# 作业题目：实现并比较 Sinusoidal Position Encoding 与 RoPE

## 要求学生完成：
1. 实现 sinusoidal position encoding；
2. 实现二维向量旋转；
3. 实现高维 RoPE；
4. 对比 E+pos 和 RoPE 的输入方式；
5. 用数值实验验证 RoPE 的相对位置性质；
6. 说明：为什么 RoPE 比简单的 E+pos 更巧妙？

## 回答：
1. Transformer 为什么需要位置编码；
2. 传统 sinusoidal position encoding 是如何生成的；
3. E + pos 的位置注入方式为什么有“内容和位置混合”的问题；
4. RoPE 不是加法，而是旋转；
5. RoPE 作用在 Q 和 K 上；
6. RoPE 的点积天然包含相对位置；
7. attention score 里的相对位置关系可以通过旋转结构自然出现。
