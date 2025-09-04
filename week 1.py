import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# 設置隨機種子以確保結果可重現
np.random.seed(0)
images = []

# 生成10張隨機的三通道影像
for i in range(10):
    color_image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    images.append(color_image)

# 創建圖表，每張圖佔一行，兩欄（左：彩色，右：灰階）
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 3))

# 用來存放每張圖的統計結果
stats_list = []

for i, color_image in enumerate(images):
    # 灰階影像（取 RGB 平均）
    gray_image = np.mean(color_image, axis=2).astype(np.uint8)

    # 左邊：彩色圖
    axes[0, i].imshow(color_image)
    axes[0, i].set_title(f'Color Image {i+1}')
    axes[0, i].axis('off')

    # 右邊：灰階圖
    axes[1, i].imshow(gray_image, cmap='gray')
    axes[1, i].set_title(f'Gray Image {i+1}')
    axes[1, i].axis('off')

    # 計算統計數據（用灰階圖）
    max_val = np.max(gray_image)
    min_val = np.min(gray_image)
    mean_val = np.mean(gray_image)
    std_val = np.std(gray_image)

    # 保存到列表
    stats_list.append([i+1, max_val, min_val, mean_val, std_val])

# 建立 DataFrame
df_stats = pd.DataFrame(stats_list, columns=['圖片編號', '最大值', '最小值', '平均值', '標準差'])

# 輸出到 Excel
output_path = os.path.join(os.getcwd(), 'test''.xlsx')
df_stats.to_excel(output_path, index=False)

print("Excel 檔案已輸出:", output_path)

plt.tight_layout()
plt.show()








