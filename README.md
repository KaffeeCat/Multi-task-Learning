# Multi-task-Learning

A PyTorch implementation of  [Multi-task learning using uncertainty to weigh losses for scene geometry and semantics](http://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html) with [torchMTL](https://github.com/chrisby/torchMTL)

![](https://pic3.zhimg.com/80/v2-f30787fffe79ad4212087bb49fccd368_1440w.png)

## Example 1 : MTL for linear regression

[[知乎]](https://zhuanlan.zhihu.com/p/474528861)
[[Jupyter notebook]](https://github.com/KaffeeCat/Multi-task-Learning/blob/main/mtl_linear_regression.ipynb)
[[Paper]](https://openaccess.thecvf.com/content_cvpr_2018/html/Kendall_Multi-Task_Learning_Using_CVPR_2018_paper.html)

Random sampling on two linear equation with observation noise σ₁=3, σ₂=0.5<br>

y₁ = w₁ x + b₁ + σ₁<br>
y₂ = w₂ x + b₂ + σ₂<br>

![](https://pic4.zhimg.com/80/v2-0172d99d00c73a9a63e0425162794b6c_1440w.png)


Training with Adam optimizer

![](https://pica.zhimg.com/80/v2-d5af1c8dd44990334d520d24e1a60411_1440w.png)

Finally, σ₁,σ₂ converges to the correct values

![](https://pic1.zhimg.com/80/v2-c33d496b8f63f33c7325a2288e7fe0ed_1440w.png)

## Example 2 : MTL for image classification

[[Jupyter notebook]](https://github.com/KaffeeCat/Multi-task-Learning/blob/main/mtl_image_classification.ipynb)

Dataset : MNIST <br>
MTL Architecture : Shared backbone and split two separated head to do fully connected forward

Training with Adam optimizer

![](https://pic3.zhimg.com/80/v2-51ca068c268d1dfeb9c1aa738da40eeb_1440w.png)

Finally, σ₁,σ₂ converges to the correct values

![](https://pica.zhimg.com/80/v2-4b4a39af8a0c20ba0f1e36dd94f07d57_1440w.png)