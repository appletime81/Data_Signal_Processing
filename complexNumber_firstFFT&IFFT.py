'''
Collated by Prof. Ching-Shih Tsou 鄒慶士 教授 (Ph.D.) at the Dept. of ME and AI&DS (機械工程系與人工智慧暨資料科學研究中心主任), MCUT(明志科技大學); the IDS (資訊與決策科學研究所), NTUB (國立臺北商業大學); the CARS(中華R軟體學會創會理事長); and the DSBA(臺灣資料科學與商業應用協會創會理事長)
Notes: This code is provided without warranty.
'''

#### A. Python的複數
# importing "cmath" for complex number operations 
# import cmath 
  
# Initializing real numbers 
x = 5
y = 3
  
# converting x and y into complex number 
z = complex(x, y); 

[(nam, type(getattr(z, nam))) for nam in dir(z)]

# printing real and imaginary part of complex number 
print ("The real part of complex number is : {}".format(z.real)) 
  
print ("The imaginary part of complex number is : {}".format(z.imag)) 

z.conjugate()

-2j # (-0-2j)

#### B. DFT與IDFT簡例(by FFT)
from scipy.fftpack import fft, ifft
import numpy as np # Why not "from numpy.fft import fft, ifft" ? Because of the conflicting namespace.

#### 假定時間序列為離散序列，從1到3中隨機選取5數
x = np.random.choice(range(1,4), 5)
# x = np.arange(5)
type(x) # numpy.ndarray
x.dtype # dtype('int64')

from matplotlib import pyplot as plt 
plt.plot(x)

# 傅立葉轉換為複數(scipy and numpy)
# by scipy
fft(x)
type(fft(x)) # numpy.ndarray
fft(x).dtype # 複數 dtype('complex128')

# by numpy
np.fft.fft(x)
type(np.fft.fft(x)) # numpy.ndarray
np.fft.fft(x).dtype # 複數 dtype('complex128')

#### 逆傅立葉轉換為原時間序列(結果為複數形式，雖然虛部為0)
# by scipy
ifft(fft(x)) # array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j])
type(ifft(fft(x)))
# The returned complex array contains ``y(0), y(1),..., y(n-1)`` where    
#    ``y(j) = (x * exp(2*pi*sqrt(-1)*j*np.arange(n)/n)).mean()``.

#### 注意！資料型別還是dtype('complex128')，雖然虛部均為0
ifft(fft(x)).dtype 
x

# by numpy
np.fft.ifft(np.fft.fft(x)) # array([0.+0.j, 1.+0.j, 2.+0.j, 3.+0.j, 4.+0.j])
type(np.fft.ifft(np.fft.fft(x)))
np.fft.ifft(np.fft.fft(x)).dtype # 還是dtype('complex128')，雖然虛部均為0

#### np.allclose()檢查逆轉換回來之時間序列是否與序列接近
np.allclose(ifft(fft(x)), x, atol=1e-15)  # within numerical accuracy. True
np.allclose(np.fft.ifft(np.fft.fft(x)), x, atol=1e-15)  # within numerical accuracy. True

# import numpy as np

# def DFT(x):
#     """
#     Compute the discrete Fourier Transform of the 1D array x
#     :param x: array
#     """
    
#     N = x.size
#     n = np.arange(N)
#     k = n.reshape((N, 1))
#     e = np.exp(-2j * np.pi * k * n / N)
#     return np.dot(e, x)