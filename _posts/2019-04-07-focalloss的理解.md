  1. CrossEntropy loss: L = -ylog(y') - (1 - y)log(1 - y'), 等价于 L = -log(y’) if y = 1; L = -log(1 - y') if y = 0
  
  2. 若考虑对正负样本给予不同的权值，即正负样本对最终loss的贡献大小不一样，可做如下处理: L = -alpha * log(y’) if y = 1; L = -(1 - alpha) * log(1 - y') if y = 0。其中alpha取值为[0,1],若alpha大于0.5,则偏向于正样本,否则偏向于负样本。
  
  3. 若再考虑对于模型对于简单样本和困难样本的区分程度，则可做进一步改进 L = -alpha * (1 - y')^gamma * log(y’) if y = 1; L = -(1 - alpha) * y'^gamma * log(1 - y') if y = 0。
  
      其中gamma > 0使得减少易分类样本的损失,使得更关注于困难的、错分的样本。
      
      若样本的真实标签为1时，当y'^趋于1时，表示这是个easy样本，需要削弱它对loss的贡献，(1 - y')^gamma就趋于0；相反，如果y'^趋于0，表示这是个hard样本，需要加强它对loss的贡献，(1 - y')^gamma就趋于1。
      
      若为负样本时有同样结论。
  
 
