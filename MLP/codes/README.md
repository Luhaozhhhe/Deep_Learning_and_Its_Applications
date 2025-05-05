Lab01:MLP

1.项目介绍

- model下的三个文件分别为我们的基础模型，调整模型和Mixer模型。本人将训练的结果存放到了results文件夹下。

- 其中，\_\_init\_\_.py文件负责初始化，并测试GPU；load_data.py负责导入我们的测试数据；mian.py是主函数文件，主要负责我们的训练；param_lr.py负责测试修改lr对结果的影响；param_optim.py负责测试修改优化模型对结果的影响；plot.py负责进行做图；train.py负责进行模型的训练。


2.项目环境配置：

python -m venv MLP

source MLP/bin/activate

pip install -r requirements.txt


3.数据集下载：
- 运行命令行：
python3 load_data.py

4.模型训练：

如果需要对模型进行训练，可以使用下面的命令行：

- 测试基础模型：
python3 main.py --model mlp_base

- 测试优化后的模型：
python3 main.py --model mlp_myself

- 测试Mixer模型：
python3 main.py --model mlp_mixer

想要测试微调参数和优化模型对结果的影响：

- 测试学习率的影响：
python3 param_lr.py

- 测试optim参数的影响：
python3 param_optim.py