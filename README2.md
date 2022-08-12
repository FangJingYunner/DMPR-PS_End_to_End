## 代码运行
- 主函数为train_end2end.py
  - --dataset_directory /media/fjy/SHARE/dataset/ps2.0/training 为数据集路径
  - --labels_directory /media/fjy/SHARE/dataset/ps2.0/ps_json_label/training 为json格式标签路径
  - --depth_factor 解码部分需要设置的特征通道深度
- 网络在model/detector的DirectionalPointDetector类中
- 网络不同深度的通道数需要修改--depth_factor参数，若编码器特征输出是1×1024×16*16,则depth_factor参数设置为32,若编码器特征输出是1×512×16*16,则depth_factor参数设置为16

