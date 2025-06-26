# 深度学习大作业

**小组成员：**

- 刘家骥 计算机科学与技术
- 陆皓喆 信息安全
- 王俊杰 计算机科学与技术
- 孙致勉 物联网工程
- 赵熙 计算机科学与技术

**选题：** 使用PyTorch框架和CPU实现CIFAR100分类模型搭建

## 使用说明

**环境配置：**

```bash
python3 -m venv CNN
source CNN/bin/activate
pip install -r requirements.txt
```

**数据集下载** ：

```bash
python3 load_data.py
```



您可以将您设计的神经网络放到`/model`文件夹下，注意命名规范（参考已有模型进行命名），并在main.py中添加部分内容。

```python
from model.CNN_Base import BaseCNN

supported_models = [
    'cnn_base',
    'cnn_resnet18',
    'cnn_resnet34',
    'cnn_resnet50',
    'cnn_resnet101',
    'cnn_resnet152',
    'cnn_densenet121',
    'cnn_densenet169',
    'cnn_densenet201',
    'cnn_densenet264',
    'cnn_se_resnet18',
    'cnn_se_resnet34',
    'cnn_se_resnet50',
    'cnn_se_resnet101',
    'cnn_se_resnet152'
]

if args.model == 'cnn_base':
    model = BaseCNN(num_of_last_layer=100)
```

上面给出了调用逻辑。我们可以使用下面的命令行来进行训练。

```bash
python3 main.py --model cnn_base
python3 main.py --model cnn_resnet18
python main.py --model convnext_tiny

```

更换后面的最后一个model参数就可以了，日志和结果会保存到我们的`results/模型名`路径中

路径中包含：

- train_log
- acc和loss的两张图像
- 训练出来的model.pth

## 任务说明

**可选论文：**

- https://openaccess.thecvf.com/content/CVPR2022/papers/Liu_A_ConvNet_for_the_2020s_CVPR_2022_paper.pdf
- https://proceedings.neurips.cc/paper_files/paper/2022/file/08050f40fff41616ccfc3080e60a301a-Paper-Conference.pdf
- https://openaccess.thecvf.com/content/ICCV2023/papers/Li_Large_Selective_Kernel_Network_for_Remote_Sensing_Object_Detection_ICCV_2023_paper.pdf
- https://proceedings.neurips.cc/paper/2021/file/20568692db622456cc42a2e853ca21f8-Paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Wang_ECA-Net_Efficient_Channel_Attention_for_Deep_Convolutional_Neural_Networks_CVPR_2020_paper.pdf
- https://openaccess.thecvf.com/content_CVPRW_2020/papers/w28/Wang_CSPNet_A_New_Backbone_That_Can_Enhance_Learning_Capability_of_CVPRW_2020_paper.pdf
- https://openaccess.thecvf.com/content_CVPR_2020/papers/Han_GhostNet_More_Features_From_Cheap_Operations_CVPR_2020_paper.pdf
- https://proceedings.neurips.cc/paper_files/paper/2022/file/436d042b2dd81214d23ae43eb196b146-Paper-Conference.pdf
- https://openaccess.thecvf.com/content/CVPR2022W/ECV/papers/Zhang_ResNeSt_Split-Attention_Networks_CVPRW_2022_paper.pdf
- https://proceedings.neurips.cc/paper/2021/file/4cc05b35c2f937c5bd9e7d41d3686fff-Paper.pdf

