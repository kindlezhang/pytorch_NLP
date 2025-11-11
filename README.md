# pytorch_NLP
This is a repository for transformers in NLP study， especially language translation.

# materials

all materials come from paoge's course


# mechanism

input是一次性输入的，但是output是不断循环基于之前的output依次输出的，output shift right给一个起始键，得到output以后再回头继续循环。

对文本进行tokenize处理的时候，先把字符串切分成token，再根据出现频次排序，再转化为one-hot variable，但是one-hot variable维度巨大。

所以我们要使用word embedding来解决one-hot的问题。词嵌入矩阵的V为词表大小，d是输入矩阵的大小，Transformer中是512. 矩阵实际上是一个权重表，经过训练后，可以将数值映射为token。最后可以将token的维度从1*d降为1*2,就可以在坐标系中直观地展示，相似语义的会聚集，差异的会分散。 

位置编码上，PE的矩阵维度是pos * d, 就是序列的长度 和 词嵌入矩阵的d。
只有维度匹配了，才能让word embedding矩阵和pos矩阵能够相加。

transformer中encoder使用的是多头自注意力，decoder中使用的是带因果掩码机制的多头自注意力，还有交叉注意力机制。

多头注意力机制中要注意的是，矩阵运算，先拼接再点积和先点积再拼接是一样的。最后用一个W0矩阵点积，使多头拼接变成融合，把维度再变回L*d。不同的头在不同的子控件与不同的相对位置上做对齐，关注的领域不同。

填充掩码：同批次中，所有句子的长度要一致来满足GPU上的并行计算。
因果掩码：就是在得分矩阵上加上一个上三角矩阵，将token后面的注意力分数变为0.

层归一化类似resnet中的批归一化，但是对象不同，因为批归一化在处理边长序列时候会受到限制，结合ppt理解一下。

推理过程是解码器逐步给出信息，结合信息内容不断循环，一个一个token得出来。（比如给一个起始命令<bos>，结合输入翻译第一个token，比如I， 结合I 和input翻译第二个词 am（得出概率最大的词）， 然后用I am 结合 input翻译 a，以此类推，每次只翻译一个token。 也需要使用mask，与训练的区别是，一个是直接输入所有的标签，一个是逐步输入推理的结果）（如何验证）
训练过程是（翻译），是根据解码器的结果，结合正确翻译的label，对每个token计算loss，反向传播更新参数。(把标签放入解码器中，通过因果掩码来防止程序知道后面的数值是什么， 比如你要翻译我是一条狗，标签是I am a dog，输入了标签I am a dog， 系统自然就知道答案了，所以在翻译是之前，只能看到I ，在翻译一只之前，只能看到I am才行)（如何训练）
总之推理过程output输入的是<bos>，训练过程输入的是完整的label。

交叉注意力机制中，本文使用的是最后一层编码器的k,v传入所有解码器中。除此以外还有多种方法。


