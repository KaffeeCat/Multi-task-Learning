# Multi-task-Learning
The study and experiment of Multi-task Learning

## Example 1 : MTL for linear regression

Random sampling on two linear equation with observation noise<br>

y₁ = w₁ x + b₁ + σ₁<br>
y₂ = w₂ x + b₂ + σ₂<br>

![](https://pic4.zhimg.com/80/v2-0172d99d00c73a9a63e0425162794b6c_1440w.png)

Define MTL tasks
```
tasks = [
        {
            'name': "InputTask",
            'layers': Sequential(*[nn.Linear(input_size, hidden_size), nn.ReLU()]),
        },    
        {
            'name': "Linear1",
            'layers': nn.Linear(hidden_size, output1_size),
            'anchor_layer': "InputTask"
        },    
        {
            'name': "Linear2",
            'layers': nn.Linear(hidden_size, output2_size),
            'anchor_layer': "InputTask"
        },
        {
            'name': "MultiLoss",
            'layers': MultiTaskLossWrapper(num_tasks=2),
            'anchor_layer': ['Linear1', 'Linear2']
        }
    ]

model = MTLModel(tasks, output_tasks=['MultiLoss'])
```

Training with Adam optimizer, loss, σ₁,σ₂ fluctuates during training

![](https://pica.zhimg.com/80/v2-d5af1c8dd44990334d520d24e1a60411_1440w.png)

![](https://pic1.zhimg.com/80/v2-c33d496b8f63f33c7325a2288e7fe0ed_1440w.png)



Jupyter notebook : [mtl_linear_regression.ipynb](https://github.com/KaffeeCat/Multi-task-Learning/blob/main/mtl_linear_regression.ipynb)

Paper : [Multi-Task Learning Using Uncertainty to Weigh Losses for Scene Geometry and Semantics](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)
