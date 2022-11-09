import numpy as np
import matplotlib.pyplot as plt
from scipy import signal




if __name__=="__main__":
    data = np.loadtxt("Angle_data2.txt")
    #data = data[:750]
    dt = 0.002
    #t_duration = np.linspace(0,0.5,2000,False)
    t = np.arange(0,1,dt)
    sign = np.sin(2*np.pi*40*t) + np.sin(2*np.pi*10*t)
    
    #fft_threashold = 5000
    #FFT
    n = len(sign)
    fhat = np.fft.fft(sign,n)
    PSD = fhat * np.conj(fhat) / n
    freq = (1/(dt*n)) * np.arange(n)
    L = np.arange(1,np.floor(n),dtype='int')

    abs_PSD = np.abs(PSD[1:len(PSD)//2])
    #PSD = PSD[1:len(PSD)//2]
    max = np.max(abs_PSD)
    #fft_threashold = max//2

    #indices = PSD > fft_threashold
    #PSD_clean = PSD * indices
    #fhat = indices * fhat
    ffilt = np.fft.ifft(fhat)

    print("Peak")
    PSD = abs(PSD)
    abs_PSD = np.abs(PSD[1:len(PSD)//2])
    #PSD = PSD[1:len(PSD)//2]
    max = np.max(abs_PSD)
    argmax = np.argmax(abs_PSD)
    print(max)
    print(str((argmax+1)*1/(dt*n)) + " hz")

    """#FFT Window
    wn = len(sign)
    window = np.hamming(n)
    windoed_sign = sign*window
    wfhat = np.fft.fft(windoed_sign,n)
    wPSD = wfhat * np.conj(wfhat) / n
    wfreq = (1/(dt*n)) * np.arange(n)
    wL = np.arange(1,np.floor(n/2),dtype='int')

    windices = wPSD > fft_threashold
    wPSD_clean = wPSD * windices
    wfhat = windices * wfhat
    wffilt = np.fft.ifft(wfhat) """

    #PLOT
    fig,axs = plt.subplots(2,1)
    plt.sca(axs[0])
    plt.plot(sign,color='blue',LineWidth=1.5,label='Signal')
    #plt.xlim(sign[0],sign[-1])
    plt.legend(loc=9 ,prop={'size': 15})

    plt.sca(axs[1])
    plt.plot(freq[L],PSD[L],color='red',LineWidth=2,label='FFT' )
    #plt.xlim(freq[L[0]],freq[L[-1]])
    plt.xlabel('Frequenz in Hz')
    plt.legend(loc=9, prop={'size': 15})


    plt.show()