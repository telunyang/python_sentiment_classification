# AIdea 電影評論情緒分類

## 競賽連結
[電影評論情緒分類](https://aidea-web.tw/topic/c4a666bb-7d83-45a6-8c3b-57514faf2901 "電影評論情緒分類")

## 作業環境:
- Windows 10 Home Edition or Linux Ubuntu 18.04+
- Anaconda (Python 3.7+)
  - [Anaconda 下載](https://www.anaconda.com/products/individual "Anaconda 下載")

## 開始前的準備與流程
- 以 桌機 / 筆電 有 GPU 的 Windows 環境為例
- 沒有 GPU，就只好 CPU Only
- 請注意 nvidia driver 與 CUDA、CUDA 與 CuDNN 之間的相依問題

## 有 GPU 的安裝流程
- 安裝 [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive "CUDA Toolkit Archive")
  - 下載 CUDA 前，請先至 Simple Transformers 及 PyTorch 網站，了解目前支援 CUDA 的版本，下載 cuDNN 亦同。
  - ![INSTALL PYTORCH](https://i.imgur.com/xBctpZ0.png "INSTALL PYTORCH")
- 安裝 [NVIDIA cuDNN](https://developer.nvidia.com/cudnn "NVIDIA cuDNN")
  - 需要先申請帳號密碼，才能進入下載頁面
  - 中間可能會請你填寫問卷，依實際情況填寫即可
- 安裝 PyTorch
  - 它使用 PyTorch 框架，所以要先了解 PyTorch 支援的 CUDA 版本: [INSTALL PYTORCH](https://pytorch.org/ "INSTALL PYTORCH")
- 安裝 Simple Transformers
  - [Simple Transformers 安裝說明](https://simpletransformers.ai/docs/installation/ "Simple Transformers 安裝說明")
- **(非必要)** 安裝 nVIDIA driver: 
  - [NVIDIA 驅動程式下載](https://www.nvidia.com.tw/Download/index.aspx?lang=tw "NVIDIA 驅動程式下載")
    - `安裝 CUDA 時，會連同 nVIDIA driver 一起安裝`，視安裝需求而定。
  - 下載方式分為「SD」與「GRD」
    - 如果需要對最新遊戲、DLC 提供最即時支援的玩家，請選擇 Game Ready 驅動程式。
    - 如果需要對影片編輯、動畫、攝影、繪圖設計和直播等創作流程提供穩定的品質，請選擇 Studio 驅動程式。
    - 如果不確定，就先選 Game Ready。

## 沒有 GPU 的安裝流程
請上 google colab 使用免費的方案，勿超過額度。如果還是沒有辦法使用 GPU，沒關係，我們可以使用 CPU 來訓練，只是會比較久…按照下列步驟安裝:
1. 一樣透過 conda 安裝 st 環境
  ```conda create -n st python=3.8 pandas tqdm```
2. 切換到 conda 環境 st
  ```conda activate st```
3. 直接安裝 PyTorch CPU Only 版本
  ```pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html```
4. 安裝 simple transformers
  ```pip install simpletransformers```
5. 訓練和預測參數都將 use_cuda 設為 **False**

## 雲端硬碟下載 CUDA 11.0 + CUDNN 11.0 for Windows 10 (之後可能會刪除):
  - [windows10_cuda110_cudnn110](https://reurl.cc/0jdmml "window10_cuda110_cudnn110")

## 檢測目前電腦有沒有支援 CUDA 的程式碼
```
import torch
print(torch.cuda.is_available())
print(torch.cuda.current_device())
print(torch.cuda.device(0))
```
```
若有支援，應該輸出類似下方的字樣:
True
0
<torch.cuda.device object at 0x0000024A91A63130>   
```

## 預訓練模型選擇
![文字理解的預訓練模式選擇](https://i.imgur.com/vvjsnZl.png "文字理解的預訓練模式選擇")

## 教學影片 (僅限學員使用，7 月中旬才開放)
拍的時候吃太飽，有些語無倫次，請見諒 Orz
[YouTube: 使用 simpletransformers，進行文字分類](https://www.youtube.com/watch?v=bzQQScSivE8 "使用 simpletransformers，進行文字分類")