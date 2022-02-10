# 實驗內容
## a.
- 需要搭配 transforms 把圖片資料轉換成 tensor 資料，且必須是 float 型態
- 套件可以使用 from torchvision.transforms import ToTensor
## b.
- 一般就 batch_size, learning_rate 可以簡單設定
- 又或是優化器或損失函數也有其他種函式可以使用
- 優化器部分有 momentum 動量參數可使用
## c.
- 先判斷 GPU 是否可用 torch.cuda.is_available()
- 然後將硬體設置 device = torch.device('cuda')
- 之後需要運算的 tensor 都可以用 to(device) 來在 GPU 中實行運算