  1. CrossEntropy loss: L = -ylog(y') - (1 - y)log(1 - y'), 等价于 L = -log(y’) if y = 1; L = -log(1 - y') if y = 0
  
  2. 若考虑对正负样本给予不同的权值，即正负样本对最终loss的贡献大小不一样，可做如下处理: L = -alpha * log(y’) if y = 1; L = -(1 - alpha) * log(1 - y') if y = 0。其中alpha取值为[0,1],若alpha大于0.5,则偏向于正样本,否则偏向于负样本。
  
  3. 若再考虑对于模型对于简单样本和困难样本的区分程度，则可做进一步改进 L = -alpha * (1 - y')^gamma * log(y’) if y = 1; L = -(1 - alpha) * y'^gamma * log(1 - y') if y = 0。
  
      其中gamma > 0使得减少易分类样本的损失,使得更关注于困难的、错分的样本。
      
      若样本的真实标签为1时，当y'^趋于1时，表示这是个easy样本，需要削弱它对loss的贡献，(1 - y')^gamma就趋于0；相反，如果y'^趋于0，表示这是个hard样本，需要加强它对loss的贡献，(1 - y')^gamma就趋于1。
      
      若为负样本时有同样结论。
  
  4. 给出一个tensorflow的focalloss实现：
  
  
  ```
  def focal_loss_sigmoid_on_2_classification(labels, logtis, alpha=0.5, gamma=2):
	"""
	description: 
		基于logtis输出的2分类focal loss计算
	计算公式：
		pt = p if label=1, else pt = 1-p； p表示判定为类别1（正样本）的概率
		focal loss = - alpha * (1-pt) ** (gamma) * log(pt)
	
	Args:
		labels: [batch_size], dtype=int32，值为0或者1
		logits: [batch_size], dtype=float32，输入为logits值
		alpha: 控制样本数目的权重，当正样本数目大于负样本时，alpha<0.5，反之，alpha>0.5。
		gamma：focal loss的超参数
	Returns:
		tensor: [batch_size]
	"""
	y_pred = tf.nn.sigmoid(logits) # 转换成概率值
	labels = tf.to_float(labels) # int -> float

	"""
	if label=1, loss = -alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred)
	if label=0, loss = - (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
	alpha=0.5，表示赋予不考虑数目差异，此时权重是一致的
	将上面两个标签整合起来，得到下面统一的公式：
		focal loss = -alpha * (1-p)^gamma * log(p) - (1-apha) * p^gamma * log(1-p)
	"""
	loss = -labels * alpha * ((1 - y_pred) ** gamma) * tf.log(y_pred) \
		-(1 - labels) * (1 - alpha) * (y_pred ** gamma) * tf.log(1 - y_pred)
	return loss

  ```
 
