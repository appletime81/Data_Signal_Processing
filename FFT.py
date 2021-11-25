'''
Collated by Ching-Shih Tsou 鄒慶士 博士 (Ph.D.) Distinguished Prof. at the Department of Mechanical Engineering/Director at the Center of Artificial Intelligence & Data Science (機械工程系特聘教授兼人工智慧暨資料科學研究中心主任), MCUT (明志科技大學); Prof. at the Institute of Information & Decision Sciences (資訊與決策科學研究所教授), NTUB (國立臺北商業大學); the Chinese Academy of R Software (CARS) (中華R軟體學會創會理事長); the Data Science and Business Applications Association of Taiwan (DSBA) (臺灣資料科學與商業應用協會創會理事長); the Chinese Association for Quality Assessment and Evaluation (CAQAE) (中華品質評鑑協會常務監事); the Chinese Society of Quality (CSQ) (中華民國品質學會大數據品質應用委員會主任委員
Notes: This code is provided without warranty.
'''

#### Complex FFT and Interpretations
from scipy.fftpack import fft, ifft
import numpy as np
import matplotlib.pyplot as plt 
np.set_printoptions(formatter={"float_kind": lambda x: "%g" % x})
fc=10 # frequency of the carrier
fs=32*fc # sampling frequency with oversampling factor=32
t=np.arange(start = 0,stop = 2,step = 1/fs) # 2 seconds duration ((2-0)/(320**(-1))=640, 2-(320**(-1))=1.996875)
x=np.cos(2*np.pi*fc*t) # time domain signal (real number)

N=256 # FFT size
X = fft(x,N) # N-point complex DFT, output contains DC at index 0
# Nyquist frequency at N/2 th index positive frequencies from
# index 2 to N/2-1 and negative frequencies from index N/2 to N-1 (Nyquist frequency included)

X[0]
abs(X[7:10])

# calculate frequency bins with FFT
df=fs/N # frequency resolution
sampleIndex = np.arange(start = 0,stop = N) # raw index for FFT plot
f=sampleIndex*df # x-axis index converted to frequencies

fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, ncols=1)
ax1.plot(t,x) #plot the signal
ax1.set_title('$x[n]= cos(2 \pi 10 t)$') 
ax1.set_xlabel('$t=nT_s$')
ax1.set_ylabel('$x[n]$')
ax2.stem(sampleIndex,abs(X),use_line_collection=True) # sample values on x-axis 
ax2.set_title('X[k]');ax2.set_xlabel('k');ax2.set_ylabel('|X(k)|'); 
ax3.stem(f,abs(X),use_line_collection=True); # x-axis represent frequencies
ax3.set_title('X[f]');ax3.set_xlabel('frequencies (f)');ax3.set_ylabel('|X(f)|');
fig.show()

nyquistIndex=N//2 #// is for integer division
print(X[nyquistIndex-2:nyquistIndex+3, None]) #print array X as column
# Note that the complex numbers surrounding the Nyquist index are complex conjugates and are present at positive and negative frequencies respectively.

#### FFT Shift
from scipy.fftpack import fftshift, ifftshift
#re-order the index for emulating fftshift
sampleIndex = np.arange(start = -N//2,stop = N//2) # // for integer division
X1 = X[sampleIndex] #order frequencies without using fftShift
X2 = fftshift(X) # order frequencies by using fftshift
df=fs/N # frequency resolution
f=sampleIndex*df # x-axis index converted to frequencies
#plot ordered spectrum using the two methods
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)#subplots creation 
ax1.stem(sampleIndex,abs(X1), use_line_collection=True)# result without fftshift 
ax1.stem(sampleIndex,abs(X2),'r',use_line_collection=True) #result with fftshift 
ax1.set_xlabel('k');ax1.set_ylabel('|X(k)|')
ax2.stem(f,abs(X1), use_line_collection=True)
ax2.stem(f,abs(X2),'r' , use_line_collection=True) 
ax2.set_xlabel('frequencies (f)'),ax2.set_ylabel('|X(f)|');
fig.show()

#### IFFTShift
X = fft(x,N) # compute X[k]
x = ifft(X,N) # compute x[n]

X = fftshift(fft(x,N)) # take FFT and rearrange frequency order
x = ifft(ifftshift(X),N) # restore raw freq order and then take IFFT

x = np.array([0,1,2,3,4,5,6,7]) # even number of elements
fftshift(x)

ifftshift(x)

ifftshift(fftshift(x))

fftshift(ifftshift(x))



x = np.array([0,1,2,3,4,5,6,7,8]) # odd number of elements
fftshift(x)

ifftshift(x)

ifftshift(fftshift(x))

fftshift(ifftshift(x))


#### Reference:
# Viswanathan, Mathuranathan, Digital Modulations using Python, December 2019.