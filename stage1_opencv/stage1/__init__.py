"""stage1_opencv：阶段 1 传统图像处理 / OpenCV 学习包。

对应学习路线（vision-ai-training 的配准、门控、合成底座）：
- io_crop_filter.py       读写图、BGR 通道序、切片裁剪、高斯/中值滤波（手写对拍）
- threshold_morphology.py 二值化（固定/大津）与形态学（腐蚀/膨胀/开闭）
- connected_components.py 连通域分析 + 外轮廓（eval 的组件 bbox+面积统计）
- transforms.py           仿射/透视变换（warp 的坐标顺序陷阱）
- registration.py         Demo A：ORB+RANSAC vs 相位相关 + 混合配准（核心）
- blending.py             Alpha 混合与羽化（Cut-Paste 的贴纸变换/融合）
- defect_pipeline.py      Demo B：经典质检流水线 + region 门控
"""
