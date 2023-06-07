# HW1  COVID-19 Cases Prediction（regression）

> Objectives:
>
> 1. Solve a regression problem with deep neural networks (DNN).
> 2. Understand basic DNN training tips.
> 3. Familiarize yourself with PyTorch.

- 一个回归模型，情景是COVID-19调查问卷，根据受访者的前两天对于问卷的填写情况以及确诊情况，预测第三天的确诊概率

  - 问卷的col有states、covid-like 症状、心理症状等等。其中states使用one-hot编码

- 此次作业的关键在于学会基础的DNN模型，全部由fully-connected-layer组成。此外获得了很多关于调参的认知

  - 使用了skl库进行feature的筛选。比起人工按照理解筛选，该库更能方便地实现feature筛选
    - feature适量为好，全选效果差
  - 参考的代码还使用了optuna进行hyper-parameter寻优，但是感觉效果不理想
    - seed、lr都算比较重要的超参
  - 更换了torch提供的optimizer，使用Adam，发现默认learning rate = 0.001 是相对最好的

- 总之是了解了DNN的基本结构，对于pytorch架构有了基本认知

  - 此外还对一些python在DL中的写法规范有了认知
  - kaggle score 在strong baseline附近，较为普通的成绩，合格地掌握HW1

- model 部分的代码

  ```python
  class My_Model(nn.Module):
      def __init__(self, input_dim):
          super(My_Model, self).__init__()
          # TODO: modify model's structure, be aware of dimensions. 
          self.layers = nn.Sequential(
              nn.Linear(input_dim, config['layer'][0]),  # 这样处理是为了更方便调参
              nn.ReLU(),   # 进行ReLU
              nn.Linear(config['layer'][0], config['layer'][1]),  # 第二层，线性组合出layer[1]个神经元
              nn.ReLU(),   # 进行ReLU
              nn.Linear(config['layer'][1], 1)  # 第三层，输出层
          )
  
      def forward(self, x):
          x = self.layers(x)
          x = x.squeeze(1) # (B, 1) -> (B)
          return x
  ```

<img src="assets/image-20230531190005821.png" alt="image-20230531190005821" style="zoom:50%;" />



# HW2 Phoneme Classification

> Objectives:
>
> 1. Solve a classification problem with deep neural networks (DNNs).
> 2. Understand recursive neural networks (RNNs).

- 一个分类模型，将语音切分为一个个小的frame，然后识别分类到41个phoneme

  - 数据预处理的部分已经由sample code 做好，只需要train model就好

  - 由于是语言问题，使用RNN的效果会好很多，此处没有使用RNN，仅训练了一个更大的DNN模型

  - model 参数情况(Total params: 1953833)

    ```python
    # data prarameters
    concat_nframes = 31              # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
    train_ratio = 0.9              # the ratio of data used for training, the rest will be used for validation
    
    # training parameters
    seed = 1213                        # random seed
    batch_size = 512                # batch size
    num_epoch = 15                   # the number of training epoch
    learning_rate = 1e-3         # learning rate
    model_path = './model.ckpt'     # the path where the checkpoint will be saved
    
    # model parameters
    input_dim = 39 * concat_nframes # the input dim of the model, you should not change the value
    hidden_layers = 6               # the number of hidden layers
    hidden_dim = 512                # the hidden dim
    ```

    

- model 部分代码

  ```python
  import torch.nn as nn
  
  class BasicBlock(nn.Module):
      def __init__(self, input_dim, output_dim):
          super(BasicBlock, self).__init__()
  
          self.block = nn.Sequential(
              nn.Linear(input_dim, output_dim),
              nn.BatchNorm1d(output_dim),
              nn.ReLU(),
              # 在此处增加 nn.Dropout()
              nn.Dropout(p=0.15)
          )
  
      def forward(self, x):
          x = self.block(x)
          return x
  
  
  class Classifier(nn.Module):
      def __init__(self, input_dim, output_dim=41, hidden_layers=4, hidden_dim=256):
          super(Classifier, self).__init__()
  
          self.fc = nn.Sequential(
              BasicBlock(input_dim, hidden_dim),
              # 这一行代码是列表推导，最终的表现是重复生成多个 hidden layer（算上上一行代码的那层）
              # 这段代码将生成hidden_layers个隐藏层
              *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers-1)],
              nn.Linear(hidden_dim, output_dim)
          )
  
      def forward(self, x):
          x = self.fc(x)
          return x
  ```

  多分类问题会在最后加一个softmax activate function，这已经被torch集成到它的crossEntrpy里面了（只要loss选crossEntrpy就会在最后加上softmax）

  

- 总之对DNN解决分类问题有了大致认知，同时该模型参数较多，约200万。若无脑再叠epoch和hidden layer的dim可能会有更好地结果。（kaggle 提供的ram 爆了

<img src="assets/image-20230531190214850.png" alt="image-20230531190214850" style="zoom:50%;" />







