'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員
Notes: This code is provided without warranty.
'''

### 繪製時域波形訊號圖
# 現在給定三個 sine 波，頻率freq與振幅amp分別為 (20, 12), (100, 5), (250, 2)，給定兩秒內取 2000 個點(每秒取樣率為 1000 Hz)後將結果繪製出來。

import numpy as np
import matplotlib.pyplot as plt
from scipy import pi

#%matplotlib inline

sample_num = 2000 # Sampling points 取樣總點數
total_time = 2 # Sampling period 取樣期間
sampling_rate = sample_num / total_time # 取樣頻率為 1000 Hz(單位時間的取樣點數為1000點)

fs = [(20, 12), (100, 5), (250, 2)] # sin 波的頻率與振幅組合。 (Hz, Amp) 頻率越來越高，振幅越來越小！
noise_mag = 2 # 雜訊變異數

time = np.linspace(0, total_time, sample_num, endpoint=False) # endpoint=False：不含右端點

vib_data = [amp * np.sin(2*pi*hz*time) for hz, amp in fs] # 公式：A*sin(2*pi*w*t)

max_time = int(sample_num / 4) # 繪圖用，只畫前500(=2000/4)點
# max_time = sample_num # 如果要畫全部的點

# 請一路全選到第二個plt.ylim((-24, 24))後執行
plt.figure(figsize=(12, 8))
# 繪製個別訊號 Show seperated signal
for idx, vib in enumerate(vib_data): # 0~2 三組正弦波
    plt.subplot(2, 2, idx+1) # 繪製idx+1個圖
    plt.plot(time[0:max_time], vib[0:max_time])
    plt.xlabel('time')
    plt.ylabel('vib_' + str(idx))
    plt.ylim((-24, 24)) # 此處先寫死

# vib加總前面三個訊號(20Hz, 100Hz, 250Hz)，並再加上雜訊
vib = sum(vib_data) + np.random.normal(0, noise_mag, sample_num) # Add noise (雜訊是從Normal(0, noise_mag = 2)中隨機取樣)
# sum(vib_data).shape # (2000,)

plt.subplot(2, 2, 4)
plt.plot(time[0:max_time], vib[0:max_time])
plt.xlabel('time')
plt.ylabel('vib(with noise)')
plt.ylim((-24, 24))

# 其中
# 左上 - 20 Hz 訊號
# 右上 - 100 Hz 訊號
# 左下 - 250 Hz 訊號
# 右下 - 前三個訊號的加總並加入雜訊的結果。

### 時域轉頻域的快速傅立葉轉換(FFT)
# 取樣時間T_{d}內取N個點，兩者相除為取樣間隔時間\delta_{t}
# T_{d}/N=\delta_{t}
# 取樣間隔時間\delta_{t}的倒數為取樣頻率F_{s}
# F_{s}=1/\delta_{t}

# 當透過FFT將資料轉換到N點的\hat{f_{n}}後，各點的間隔頻率為\delta_{F}
# \delta_{F}=F_{s}/N

# 每個點對應到的頻率為
# F_{n}=n \delta_{F}, n=0,1,...N-1

# Nyquist Frequency: 超過F_{s}/2會產生混疊(aliasing)的現象。因此 FFT 轉換後的\hat{f_{n}}只有在 0≤n≤N/2 的範圍內有意義

# 載入傅立葉轉換模組
from scipy.fftpack import fft

fd = np.linspace(start=0.0, stop=sampling_rate, num=int(sample_num), endpoint=False) # 0到1000的區間產生2000個頻率點

# 總和訊號傅立葉轉換
vib_fft = fft(vib) # Array of Complex128 (2000,) 有實有虛了！與vib維數相同
mag = 2/sample_num * np.abs(vib_fft) # Magnitude (再確認一下此公式！Since the calculated FFT will be symmetric such that the magnitude of only the first half points are unique (and the rest are symmetrically redundant), the main idea is to display only half of the FFT spectrum. )

plt.plot(fd[0:int(sample_num/2)], mag[0:int(sample_num/2)])
plt.xlabel('Hz')
plt.ylabel('Mag')

# 從上圖可以看出，在 20 Hz, 100 Hz 以及 250 Hz 的位置各有一個明顯的能量高峰，資料量與數值與之前設定的頻率一致，以此可知 FFT 可幫助我們分解找出原始訊號中的訊號組合。

### 逆快速傅立葉轉換(Inverse FFT)再轉回時域訊號
from scipy.fftpack import ifft

vib_back = ifft(vib_fft) # ifft結果vib_back(振幅 + 相位角j)，也是Array of Complex128 (2000,)

vib_re = np.real(ifft(vib_fft)) # 取實部(振幅) Real part of complex number

vib_im = np.imag(ifft(vib_fft)) # 取虛部(相位角)

plt.plot(time[0:max_time], vib_re[0:max_time]) # 轉換回來的訊號幾乎相等於原始訊號，為何只繪實部？Ans.實部是原時域訊號值
plt.ylim((-24, 24))

# # np.abs()計算振幅
# mag = np.abs(ifft(vib_fft))

# plt.plot(time[0:max_time], mag[0:max_time])
# plt.ylim((0, 24))

### 總結 (可以先看這裡)
# Fourier Series -> Complex Fourier Series (Euler's Formula) -> (因為取樣) Discrete Time Fourier Transform (矩陣形式, 反矩陣) -> Fast Fourier Transform (time complexity from O(N^2) to O(NlogN)) -> (與程式碼相關) T/N = delta_t, F = N/T = 1/delta_t -> delta_F = F/N = 1/T (N點間隔頻率), F_n = n delta_F, n = 0, 1, ..., N-1 -- Nyquist Frequency --> F/2 for aliasing 0 <= n <= N/2

### 參考資料 
# 從傅立葉級數到快速傅立葉轉換 http://blog.yeshuanova.com/2019/04/fft_intro/
