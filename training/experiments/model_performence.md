# baseline的基准训练结果
## 0. 原始H，L图像的基准指标
H：原始64bit SAR图像
L：双向上采样R2A2，RT阈值的单比特SAR结果
用的和训练代码一致的training\basicsr\metrics\psnr_ssim.py中的函数`calculate_psnr`和`calculate_ssim` 计算
**PSNR** = 26.0884  **SSIM** = 0.8024
## 1. seed 42

| 模型          | PSNR    | SSIM*  |
| ----------- | ------- | ------ |
| **HINet**   | 27.2454 | 0.8248 |
| **KBNet_s** | 27.1405 | 0.8236 |
| **NAFNet**  | 27.1306 | 0.8236 |
| **SCUNet**  | 27.0869 | 0.8211 |
