# HW01  COVID-19 Cases Prediction（regression）

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



# HW02 Phoneme Classification

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





# HW03 Image Classification

> Objectives:
>
> 1. Use CNN for image classification.
> 2. Implement data augmentation
> 3. Visualize the learned visual representations of the CNN model on the validation set by implementing t-SNE
> 4. Implement Cross Validation \+ Ensemble

- 将事物的图片分成十一类，使用CNN模型

  ```python
  class Classifier(nn.Module):
     def __init__(self):
         super(Classifier, self).__init__()
         # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
         # torch.nn.MaxPool2d(kernel_size, stride, padding)
         # input 維度 [3, 128, 128]
         self.cnn = nn.Sequential(
             nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128] 其中3,1,1分别指的kernel大小，stride，padding
             nn.BatchNorm2d(64),
             nn.ReLU(),
             nn.MaxPool2d(2, 2, 0),      # [64, 64, 64] 
            
  
             nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
             nn.BatchNorm2d(128),
             nn.ReLU(),
             nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]
           
             nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
             nn.BatchNorm2d(256),
             nn.ReLU(),
             nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]
  
             nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
             nn.BatchNorm2d(512),
             nn.ReLU(),
             nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
             
             nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
             nn.BatchNorm2d(512),
             nn.ReLU(),  
             nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
         )
         self.fc = nn.Sequential(
             nn.Dropout(0.4),
             nn.Linear(512*4*4, 1024),
             nn.ReLU(),
             nn.Linear(1024, 512),
             nn.ReLU(),
             nn.Linear(512, 11)
         )
  
     def forward(self, x):
         out = self.cnn(x)
         out = out.view(out.size()[0], -1)
         return self.fc(out)
  ```

  

- 通过 `torchvision.transforms` 实现data augmentation

  ```python
  train_tfm = transforms.Compose([
      transforms.Resize((128, 128)),
      transforms.RandomHorizontalFlip(0.5),
      transforms.RandomVerticalFlip(0.5),
      transforms.RandomRotation(50),
      transforms.GaussianBlur(3, 0.1),
      transforms.ColorJitter(brightness=0.5, hue=0.3),
      transforms.ToTensor(),
      # ToTensor() should be the last one of the transforms.
  ])
  ```

  [ CSDN：torchvision.transforms 常用方法解析](https://blog.csdn.net/weixin_42426841/article/details/129903800)

  augmentation效果：<img src="assets/image-20230608143505567.png" alt="image-20230608143505567" style="zoom:50%;" />

- 通过t-SNE降维可视化解释模型分类效果

  **top层（FC层前的最后一层）**

  <img src="assets/image-20230608133750969.png" alt="image-20230608133750969" style="zoom:50%;" />

  **mid层（CNN中的中间层）**

  <img src="assets/image-20230608135838540.png" alt="image-20230608135838540" style="zoom:50%;" />

  可见top layer 比 mid layer 的分类效果明显很多，top层中，相同颜色的点代表相同类的样本，冒险可以看出有聚集的趋势，并且可以通过两类样本或个别样本的距离解释相似程度。

- 使用k - fold cross  validation \+ Ensemble 可以提高模型的稳定性。此HW中我使用了4折，train出了4个model，最终是使用的简易的vote机制来确定结果。由于模型对每个样本给出的一个概率分布序列，我采取的是将4个分布叠加起来后再取分布中的最大值当做prediction。最后的结果并不理想，在于每个model的结果都较差，ensemble并没有得到很好的体现，但是由于train的时间太长了，只跑了15个epoch就要2h，4-fold 就是4倍的时间。最终放弃刷榜。

  <img src="assets/image-20230608144511793.png" alt="image-20230608144511793" style="zoom:50%;" />
