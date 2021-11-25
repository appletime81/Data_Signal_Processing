'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員
Notes: This code is provided without warranty.
'''

# Take an arbitrary cosine function of the form x(t) = Acos(2π fct +φ) and proceed step by step as follows
# • Represent the signal x(t) in computer memory (discrete-time x[n]) and plot the signal in time domain
# • Represent the signal in frequency domain using FFT (X[k])
# • Extract magnitude and phase information from the FFT result
# • Reconstruct the time domain signal from the frequency domain samples

from scipy.fftpack import fft, ifft, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt
A = 0.5 # amplitude of the cosine wave
fc = 10 # frequency of the cosine wave in Hz
phase = 30 # desired phase shift of the cosine in degrees
fs = 32*fc # sampling frequency with oversampling factor 32
t = np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration
phi = phase*np.pi/180; # convert phase shift in degrees in radians 轉換為弳/弧度量
x = A*np.cos(2*np.pi*fc*t+phi) # time domain signal with phase shift

N = 256 # FFT size
X = 1/N*fftshift(fft(x, N)) # N-point complex DFT

df = fs/N # frequency resolution
sampleIndex = np.arange(start = -N//2, stop = N//2) # // for integer division (-128, 127)
f = sampleIndex*df # x-axis index converted to ordered frequencies

fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=4, ncols=1)
ax1.plot(t,x) # plot time domain representation 
ax1.set_title(r'$x(t) = 0.5 cos (2 \pi 10 t + \pi/6)$')
ax1.set_xlabel('time (t seconds)');ax1.set_ylabel('x(t)')


ax2.stem(f, abs(X), use_line_collection=True) # magnitudes vs frequencies
ax2.set_xlim(-30, 30)
ax2.set_title('Amplitude spectrum')
ax2.set_xlabel('f (Hz)');ax2.set_ylabel(r'$ \left| X(k) \right|$')

phase=np.arctan2(np.imag(X),np.real(X))*180/np.pi # phase information
ax3.plot(f,phase) # phase vs frequencies

X2 = X #store the FFT results in another array
# detect noise (very small numbers (eps)) and ignore them
threshold = max(abs(X))/10000; # tolerance threshold
X2[abs(X)<threshold]=0 # maskout values below the threshold 
phase=np.arctan2(np.imag(X2),np.real(X2))*180/np.pi # phase information
ax4.stem(f,phase, use_line_collection=True) # phase vs frequencies
ax4.set_xlim(-30, 30);ax4.set_title('Phase spectrum')
ax4.set_ylabel(r"$\angle$ X[k]");ax4.set_xlabel('f(Hz)')
fig.show()

X[0:5]

np.arctan2(np.imag(X[0:5]),np.real(X[0:5]))

#### Reconstructing the time domain signal from the frequency domain samples
x_recon = N*ifft(ifftshift(X),N) # reconstructed signal
t = np.arange(start = 0,stop = len(x_recon))/fs # recompute time index

fig2, ax5 = plt.subplots()
ax5.plot(t,np.real(x_recon)) # reconstructed signal
ax5.set_title('reconstructed signal')
ax5.set_xlabel('time (t seconds)');ax5.set_ylabel('x(t)');
fig2.show()
 
#### Reference:
# Viswanathan, Mathuranathan, Digital Modulations using Python, December 2019.