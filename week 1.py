import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 設置隨機種子以確保結果可重現
np.random.seed(0)
images = []

# 生成10張隨機的三通道影像
for i in range(10):
    color_image = np.random.randint(0, 256, size=(32, 32, 3), dtype=np.uint8)
    images.append(color_image)

# 創建圖表，每張圖佔一行，兩欄（左：彩色，右：灰階）
fig, axes = plt.subplots(nrows=2, ncols=10, figsize=(15, 3))

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

print("Mean:",np.mean(images))
print("Max",np.max(images))
print("Min",np.min(images))
print("Std:",np.std(images))

# 生成線性數列
sequence_1 = np.linspace(1, 3, 10)

# 建立字典資料
d = {'s1': sequence_1}

# 建立 DataFrame
df = pd.DataFrame(data=d)

# 加入統計列
stats = pd.DataFrame({
    's1': [np.max(df['s1']), np.min(df['s1']), np.mean(df['s1']), np.std(df['s1'])]
}, index=['Max', 'Min', 'Mean', 'Std'])

# 把原始資料與統計數據合併（index 要重設）
df_combined = pd.concat([df, stats])

# 輸出到 Excel
df_combined.to_excel('test.xlsx')

plt.tight_layout()
plt.show()








