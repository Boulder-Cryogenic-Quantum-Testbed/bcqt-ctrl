import glob, csv
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt
plt.rcParams['savefig.facecolor']='white'

def loadVNA(FileList):
    """
    The data structure anticipated is single line header of columns with columns units in freq [Ghz], Power [dB], Phase [rad].
    The data phase will be corrected without request if the phase exceeds 10 'degree' into radians.  
    Input: 
        FileList, list of full path file names to csv data
    Output:
        data, array of data in freq [Ghz], Power [dB], Phase [rad] with the phase unwrapped
    
    """
    data = dict()
    column = ['Freq_Hz', 'Amp_dB', 'Phase_Rad']
    for file in FileList:
        key = file.split('/')[-1].replace('_merged.csv', '')
        data[key] = dict()
        with open(file) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            host=[row for row in csv_reader]
        start = 0
        while True:
            try:
                float(host[start][0])
                break
            except:
                start+=1
            if 10<=start:
                print('Starting point is not in first 10 rows')
                break
        for ll, line in enumerate(np.transpose(host[start:]).astype(float)):
            if (ll==2) & (10<=np.max(np.abs(line))):
                line=line*np.pi/180 # Change phase units to radians
            if (ll==2):
                line = np.unwrap(line)
            data[key][column[ll]]=line
    return data

def pulse_filter(yaxis, gap = 25, length = 55):
    """
        Takes the y values for a trace and applies a trapazoidal filter in units of array index:
        Example:
        taxis  = np.arange(16)
        yaxis  = np.array([0]*4 + [1]*8 + [0]*4)
        output = pulse_filter(yaxis, 5, 4)
        output: [0, 0, 0, 0, 0, 0, 0.75, 0.25, -0.25, -0.75, 0, 0, 0, 0, 0, 0]
    """
    if gap%2 == 1: #Gap is odd number of points
        pass
    else:
        print('Gap must be odd')
        raise
    pfilter = np.zeros(2*length + gap)
    pfilter[:length] = 1/length
    pfilter[-length:] = -1/length

    out = np.zeros(len(yaxis))
    if 500 < len(yaxis):
        out[length + int((gap-1)/2):-(length + int((gap-1)/2))] = np.convolve(yaxis, pfilter, 'valid')
    else:
        out[length + int((gap-1)/2):-(length + int((gap-1)/2))] = sig.fftconvolve(yaxis, pfilter, 'valid')
    return out

def find_tones(faxis, Aaxis, Paxis, partition = 0.1, sigma = 5, verbose = True):
    """
    faxis, array of the scanned frequency in GHz
    Aaxis, array of the magnitude of the S21 in dB
    Paxis, array of the phase of the S21 in radian and already unwrapped
    partition, float indicating the width over which to break up the data set for analysis in GHz.
    sigma, float indicating the number of STD from the maximum partition fluctuations 
    """
    # Apply two trapazoidal filters to the data to clean up the noise and long scale features from impedance mismatches.
    zaxis=pulse_filter(pulse_filter(Aaxis))
    paxis=pulse_filter(pulse_filter(Paxis))

    # Calculate histogram and get the fwhm of the 'gaussian' baseline, which is not exactly gaussian
    # Data is broken into 0.1 GHz steps to determine the statistical fluctuations from zero
    # for each inteval, the maximum deviation from the mean is taken as the threshold for that 
    # interval. This threshold is scaled by sigma which if it was a standard deviation would exclude 
    # nearly all the random flucuations. The final threshold values for each interval is taken to
    # be the maximum threshold of nearest neighbors. Note that the first and last intervals only
    # have one nearest neigbhor to check.
    ythresh=[]
    edges=np.round([faxis[0], faxis[-1]], 3)
    edges=np.linspace(edges[0], edges[1], int((edges[1]-edges[0])/partition)+1)
    for ee, elem in enumerate(edges[1:]):
        tag = (edges[ee]<=faxis)&(faxis<=elem)&(zaxis!=0)
        nspec, bins = np.histogram(zaxis[tag], bins = 1000, range = (-1, 1))
        bmid = (bins[1:]+bins[:-1])/2
        yt=bmid[0.5*nspec.max()<=nspec]
        ythresh.append(np.abs(yt[::yt.size-1]).max())
    ythresh=sigma*np.array(ythresh)
    erevise=np.transpose([edges[:-1], edges[1:]])
    fthresh=np.zeros(ythresh.size)
    fthresh[0]=ythresh[:2].max()
    fthresh[-1]=ythresh[-2:].max()
    fthresh[1:-1]=np.array([ythresh[:-2], ythresh[1:-1], ythresh[2:]]).T.max(axis=1)

    # Using the finalized thresholds perform a peak search in all areas and save the location 
    # of the features.
    feature=[]
    for ee, elem in enumerate(edges[1:]):
        tag = (edges[ee]<=faxis)&(faxis<=elem)
        lpeak = find_peaks(zaxis[tag], height = fthresh[ee], distance=1000)[0]
        if lpeak.size!=0:
            feature.append([np.arange(faxis.size)[tag][lpeak], faxis[tag][lpeak]])
    feature=np.hstack(feature)
    print('Order\tIndex\tFreq [GHz]')
    for num, point in enumerate(feature.T):
        print(f"{num}\t{int(point[0])}\t{np.round(point[1], 5)}")
    return feature, erevise, fthresh

def plot_tones(faxis, Aaxis, Paxis, feature, erevise, fthresh, span = 0.050, gap = 25, length = 55):
    """
    faxis, array of the scanned frequency in GHz
    Aaxis, array of the magnitude of the S21 in dB
    Paxis, array of the phase of the S21 in radian and already unwrapped
    feature, 2-D array of features identified as potential resonator tones
    span, float this is th eplotting range for the features in GHz.
    gap, integar value used to filter data to identify features
    length, integar value used to filter data to identify features
    """
    # Apply two trapazoidal filters to the data to clean up the noise and long scale features.
    zaxis=pulse_filter(pulse_filter(Aaxis))
    paxis=pulse_filter(pulse_filter(Paxis))
    
    # Plot each feature identified in the prior cell
    for ff, feat in enumerate(feature.T):
        if feat[0]<length+gap/2: # This skips the point associated with the filter length
            continue
        fig, (ax, bx, cx)= plt.subplots(1, 3, figsize=(16,6), sharex = True, constrained_layout = True)
        tag = (feat[1]-span<=faxis)&(faxis<=feat[1]+span)
        stag= (feat[1]-0.005<=faxis)&(faxis<=feat[1]+0.005)
        ax.plot(faxis[tag], Aaxis[tag], 'b')
        ax.plot(faxis[stag], Aaxis[stag], 'k', linewidth=2)
        cx.plot(faxis[tag], paxis[tag], 'g')
        bx.plot(faxis[tag], zaxis[tag], 'b')
        ran = erevise[(erevise.T[0]<feat[1]) & (feat[1]<erevise.T[1])][0]
        bx.hlines(fthresh[(erevise.T[0]<feat[1]) & (feat[1]<erevise.T[1])], ran[0], ran[-1], color = 'r', label='Threshold')
        xlabels=[feat[1]-span, feat[1], feat[1]+span]
        ax.set_xlim(feat[1]-span, feat[1]+span)
        ax.set_xticks([feat[1]-0.75*span, feat[1], feat[1]+0.75*span])
        ax.set_ylabel('|S$_{21}$| [dB]', size = 16)
        ax.set_ylim(np.floor(Aaxis[tag].min()), np.ceil(Aaxis[tag].max()))
        if Aaxis[tag].max() - Aaxis[tag].min() < 4:
            ax.set_yticks(np.arange(np.floor(Aaxis[tag].min()), np.ceil(Aaxis[tag].max())+1, 1))
        ax.tick_params(labelsize=16)
        bx.set_ylabel('Trap(|S$_{21}$|) [Arb. Units]', size = 16)
        bx.set_xlabel('Frequency [GHz]', size=16)
        bx.set_xticks([feat[1]-0.75*span, feat[1], feat[1]+0.75*span])
        bx.tick_params(labelsize=16)
        bx.set_title(f'Feature at {np.round(feat[1],5)} GHz', size = 16)
        bx.legend()
        cx.set_ylabel(r'Trap($\angle$S$_{21}$) [Arb. Units]', size = 16)
        cx.set_xticks([feat[1]-0.75*span, feat[1], feat[1]+0.75*span])
        cx.tick_params(labelsize=16)
        plt.show()
        plt.close('all')
    return None

if __name__ == "__main__":
    #change paths here to direct you to your data set for testing purposes.
    run = 65
    path = '/mnt/c/Users/ponc892/OneDrive - PNNL/Documents/PNNL/BlueFors-Operations/'
    path = [line for line in glob.glob(path+'*') if f'Run_{run}' in line][0]+'/'
    path = path+'Processed/Merged_Files'

    grandparent = glob.glob(path+'/*')[-3:-2]
    host=loadVNA(grandparent)

    host = host[list(host.keys())[0]]
    faxis=1e-9*host['Freq_Hz']
    Aaxis=host['Amp_dB']
    Paxis=host['Phase_Rad']

    fig, (ax, bx)= plt.subplots(2, 1, figsize=(8,12))
    ax.plot(faxis, Aaxis, color='tab:blue')
    bx.plot(faxis, Paxis-np.polyval(np.polyfit(faxis, Paxis, 1), faxis), color='tab:green')
    ax.set_ylabel('|S$_{21}$| [dB]', size=16)
    ax.tick_params(labelsize=16)
    bx.set_ylabel(r'$\angle$S$_{21}$ [rad]', size=16)
    bx.set_xlabel('Frequency [GHz]', size=16)
    bx.tick_params(labelsize=16)
    plt.show()
    
    feature, erevise, fthresh = find_tones(faxis, Aaxis, Paxis, partition = 0.1, sigma = 5)
    plot_tones(faxis, Aaxis, Paxis, feature, erevise, fthresh, span = 0.050, gap = 25, length = 55)