'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員
Notes: This code is provided without warranty.
'''

#### Real Values Signals 實數值訊號
# CT signal
import numpy as np
import matplotlib.pyplot as plt
t = np.linspace(-0.02, 0.05, 1000)
y = 325 * np.sin(2*np.pi*50*t) # y = Amp * sin(2pi*f*t + 0)
# 高階繪圖(high-level plotting)
plt.plot(t, y); # (time, amplitude) --DFT(FFT)--> (freq, amplitude or power)
# 低階繪圖(low-level plotting)
plt.xlabel('time');
plt.ylabel('y=x(t) amplitude');
plt.title(r'Plot of CT signal $y=x(t)=325 \sin(2\pi 50 t)$');
plt.xlim([-0.02, 0.05]);
plt.show()

# DT signal (Stem Plot or histogram plot)
n = np.arange(50); # 樣本點數
dt = 0.07/50 # (end - start)/# points
x = np.sin(2 * np.pi * 50 * n * dt)
# 高階繪圖(high-level plotting)
plt.stem(n, x);
# 低階繪圖(low-level plotting)
plt.xlabel('n');
plt.ylabel('x[n]');
plt.title(r'Plot of DT signal $x[n] = 325 \sin(2\pi 50 n \Delta t)$');


#### Complex Valued Signals 複數值訊號
# x(t)=exp(j100πt), j is the complex notation in Python
# Real part and complex part
t = np.linspace(-0.02, 0.05, 1000)
plt.subplot(2,1,1); # 圖面切分：兩rows，一column，繪1st子圖
plt.plot(t, np.exp(2j*np.pi*50*t).real);
plt.xlabel('t');
plt.ylabel('Re x(t)');
plt.title(r'Real part of  $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);

plt.subplot(2,1,2); # 圖面切分：兩rows，一column，繪2nd子圖
plt.plot(t, np.exp(2j*np.pi*50*t).imag);
plt.xlabel('t');
plt.ylabel('Im x(t)');
plt.title(r'Imaginary part of  $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.show()

# 複數訊號計算
np.exp(2j*np.pi*50*t)

# Magnitude and phase 複數訊號的振幅與相位角
t = np.linspace(-0.02, 0.05, 1000)
plt.subplot(2,1,1); 
plt.plot(t, np.abs(np.exp(2j*np.pi*50*t)) );
plt.xlabel(r'$t$');
plt.ylabel(r'$|x(t)|$');
plt.title(r'Absolute value  of  $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.subplot(2,1,2); 
plt.plot(t, np.angle(np.exp(2j*np.pi*50*t))*360/(2*np.pi) );
plt.xlabel('$t$');
plt.ylabel(r'$\angle x(t)$');
plt.title(r'Phase of  $x(t)=e^{j 100 \pi t}$');
plt.xlim([-0.02, 0.05]);
plt.show()

#### References:
# https://staff.fnwi.uva.nl/r.vandenboomgaard/SP20162017/SystemsSignals/plottingsignals.html
