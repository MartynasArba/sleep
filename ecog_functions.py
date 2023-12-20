import numpy as np
import pandas as pd
from scipy import signal
import pickle
#inductive needed later
from sklearn.base import BaseEstimator, clone
from sklearn.cluster import AgglomerativeClustering
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted
# import chardet

def read_txt(sel_paths):
    """
    Reads multiple recordings and returns them in a single dataframe, with a datetime column.
    No conversion is performed. Values will still be in 6 bit hex binary twos complement.
    Params:
    sel_paths - selected paths to files
    Returns:
    data - all files combined into a dataframe.     
    """
    dfs=[]

    for path in sel_paths:
        print(path)
        start_time=pd.read_table(path, sep=' ',header=None, skiprows=2, nrows=1)[1]
        rec_year=str(path[-16:-12])
        rec_month=str(path[-10:-8])
        rec_day=str(path[-12:-10])
        rec_date=pd.to_datetime(rec_year+'-'+rec_month+'-'+rec_day+' '+str(start_time[0]))
        #get encoding
        # with open(path, 'rb') as f:
        #     enc = chardet.detect(f.readline())  # or readline if the file is large
        # print(enc['encoding'])
        df=pd.read_csv(path, sep=' ',header=None, skiprows=12)#, encoding='ascii')#, on_bad_lines='skip', encoding_errors='ignore')#, engine ='python')# encoding=enc['encoding'])
        df=df.dropna()
        df['time']=pd.date_range(rec_date, periods=len(df),freq='ms')
        dfs.append(df)
        # b.to_csv(pickle_paths[50].strip('.pkl')+'.txt')
    data=pd.concat(dfs).reset_index(drop=True)
    return data

#conversion func
def _twos_comp(val, bits=24):
    """
    INTERNAL
    Computes the two's complement of a given binary value
    Params:
    val - binary number
    bits - how many bits encode the value
    Returns:
    val - two's complement of the primary value
    """
    if (val & (1 << (bits - 1))) != 0:
        val = val - (1 << bits)        
    return val

def _hex_to_v(hex_str, gain=1, vref=4.5, bits=24):
    """
    INTERNAL
    Converts a recorded hex value to its binary twos complement, then to voltage. 
    The conversion to voltage follows the formula provided by microcontroller ADS1299: V = 1 LSB = (2 × VREF / Gain) / 2**24
    Params:
    hex_str - recorded hex value
    gain - gain of the equipment (preset to 1, adjustable by microcontroller)
    vref - reference voltage of equipment (default 4.5, adjustable by microcontroller)
    gauna vertę vref matavimo vienetu iš hex skaičiaus
    Returns:
    v - decoded voltage
    """
    out=_twos_comp(int(hex_str, 16))
    v=(((2*vref)/gain)/(2**bits))*out
    return v

def convert_to_mv(data):
    """
    Applies the internal conversion function (to number, then to volts) to four columns of a dataframe.
    Params:
    data - the unconverted data
    Returns:
    data - converted data    
    """
    data[0]=data[0].apply(_hex_to_v)
    data[1]=data[1].apply(_hex_to_v)
    data[2]=data[2].apply(_hex_to_v)
    data[3]=data[3].apply(_hex_to_v)
    data=data.rename(columns={0:'l_ecog',1:'r_ecog',2:'lfp',3:'emg'})
    return data

#converts selected channel/feature to power
def signal_to_power(signal, window=1000, rolling=False):
    """Converts selected signal to power. Will be used to transform features.

    Args:
        signal (numpy array): signal to transform
        window (int, optional): power is calculated as average - how big should the window be. Defaults to 1000.
        rolling (bool, optional): whether to use the step of window or 1. Defaults to False.

    Returns:
        power: calculated power
        indices: indices of original signal
        var: variance of subset
    """
    power = np.array([])
    indices = np.array([])
    variances = np.array([])
    if rolling == True:
        for index in range(0,len(signal)-window):
            signal_subset = signal[index: index+window]
            power_subset = ((signal_subset**2).sum())/len(signal_subset)
            power = np.append(power, power_subset)
            indices=np.append(indices,signal_subset.index[0])
            variances=np.append(variances, np.var(signal_subset))
    elif rolling == False:
        for index in range(0,int(np.floor(len(signal)/window))-1):
            signal_subset = signal[index*window: (index+1)*window]
            power_subset = ((signal_subset**2).sum())/len(signal_subset)
            power=np.append(power,power_subset)
            indices=np.append(indices, signal_subset.index[0])
            variances=np.append(variances, np.var(signal_subset))
            
    return power, indices, variances

def get_harmonic_params(x, window=1000):
    """Calculates harmonic parameters 
        From: EEG feature extraction for classification of sleep stages
        Authors: E. Estrada; H. Nazeran; P. Nava; K. Behbehani; J. Burk; E. Lucas
        Year: 2004
        doi: 10.1109/IEMBS.2004.1403125

    Args:
        x : signal to extract params from
        window (int, optional): what window size to use. Defaults to 1000.

    Returns:
        center_freqs, bandwiths, center_pows: as calculated from the formulas.
    """
    center_freqs=np.array([])
    bandwidths=np.array([])
    center_pows =np.array([])
    
    for t in range(0,int(np.floor(len(x)/window))-1):
        f, pxx = signal.periodogram(x[t*window:(t+1)*window], fs=1000, scaling='spectrum')
        freq_pows=np.array([])
        #calculating f*P
        for index in range(len(f)):
            freq_pows=np.append(freq_pows,f[index]*pxx[index])
            
        #center frequency
        fc=np.sum(freq_pows)/np.sum(pxx)
        
        #bandwidth
        bw=np.sqrt(np.sum((((f-fc)**2)*freq_pows))/np.sum(freq_pows))
        
        #value at central frequency
        p_fc = f[np.abs(f-fc).argmin()]
        
        bandwidths=np.append(bandwidths, bw)        
        center_freqs=np.append(center_freqs, fc)
        center_pows=np.append(center_pows, p_fc)
        
    return center_freqs, bandwidths, center_pows

#function to center and filter the data
def filter_channel(channel, fstart=0.1, fstop=45, sr=1000, center=True, notch_50=False):
    """A function to perform notch filtering, mean centering and bandpass filtering for a single recorded channel. 
    
    Args:
        channel (pandas series/column): A single column from the recording.
        fstart (float, optional): The low end of the range to return. Defaults to 0.1.
        fstop (int, optional): The high end of the range to return. Defaults to 45.
        sr (int, optional): Sampling rate of the recording. Defaults to 1000.
        center (bool, optional): Should centering be performed. Defaults to True.
        notch_50 (bool, optional): Should a 50 Hz Notch filter be applied. Defaults to False, because by default 50 Hz is already out of range.

    Returns:
        Output of the filtering function: a filtered channel.
    """
    if center==True:
        channel=channel-channel.mean()
        
    if notch_50==True:
        b0, a0=signal.iirnotch(50, Q=30, fs=sr)
        channel=signal.filtfilt(b0, a0, channel)
    
    sos_bandpass=signal.butter(N=20, Wn=(fstart, fstop), btype='bp', output='sos',  fs=sr)
    
    return signal.sosfiltfilt(sos_bandpass, channel)   

def warn_by_ch(data, thr_ecog=0.02, thr_emg=0.002, thr_lfp=0.003):
    """
    Prints a warning if the standard deviation is similar to noise.
    Params:
    data - filtered recording
    thr_ecog - threshold of warning for ecog
    thr_emg - threshold of warning for emg
    thr_lfp - threshold of warning for lfp
    Returns:
    damage - the number of channels with warnings
    """
    damage=0
    if data['l_ecog'].std()>thr_ecog:
        print('l_ecog may be damaged or disconnected')
        damage+=1
    if data['r_ecog'].std()>thr_ecog:
        print('r_ecog may be damaged or disconnected')
        damage+=1
    if data['emg'].std()<thr_emg:
        print('emg may be damaged or disconnected')
        damage+=1
    if data['lfp'].std()<thr_lfp:
        print('lfp may be damaged or disconnected')
        damage+=1

    return damage

def get_meanpows(data):
    """
    Calculates mean power in each frequency band for each second. 
    Params:
    data - filtered ECoG + EMG data
    Returns:
    meanpows - mean powers of each frequency interval every second    
    """
    times=np.floor(int(np.floor(len(data)/1000))/1)
    #mean power list
    areas=data.columns.drop('time')
    params=['time','delta','theta','alpha','beta','gamma','mean']
    #create df
    tuples=[]
    for area in areas:
        for param in params:
            col_name=(area, param)
            tuples.append(col_name)
    meanpows=pd.DataFrame(index=np.arange(0,times), columns=pd.MultiIndex.from_tuples(tuples))
    #fill df
    for i in range(0, int(times)):
        for area in areas:
            f, Pxx=signal.welch(data[area].iloc[i*1000:(i+1)*1000], fs=1000, window='hann', nperseg=1000, scaling='spectrum')
            meanpows[area,'time'][i]=data['time'][i*1000]
            meanpows[area,'delta'][i]=np.mean(Pxx[np.where(f<=4.5)])
            meanpows[area,'theta'][i]=np.mean(Pxx[np.where((f>4.5)&(f<=8))])
            meanpows[area,'alpha'][i]=np.mean(Pxx[np.where((f>8)&(f<=12))])
            meanpows[area,'beta'][i]=np.mean(Pxx[np.where((f>12)&(f<=20))])
            meanpows[area,'gamma'][i]=np.mean(Pxx[np.where(f>20)])
            meanpows[area,'mean'][i]=np.mean(Pxx)

    return meanpows

def normalize_meanpows(meanpows):
    """
    Normalizes mean powers by dividing each frequency band mean power from the total of all frequency bands powers.
    Params:
    meanpows - mean powers for each frequency band each second.
    Returns:
    meanpows_norm - normalized mean powers for each frequency band each second.
    """
    meanpows_norm=pd.DataFrame(0,columns=meanpows.columns, index=meanpows.index)
    for area in np.unique(meanpows.columns.get_level_values(0)):
        params=np.unique(meanpows.columns.get_level_values(1).drop(['time','mean']))
        for param in params:
            meanpows_norm[area,param]=meanpows[area,param]/meanpows[area][params].sum(axis=1)
        meanpows_norm[area,'mean']=meanpows[area,'mean']
        meanpows_norm[area,'time']=meanpows[area,'time']
    return meanpows_norm

def get_x_for_clusters(meanpows_norm):
    """
    Generates a numpy array for clustering later. 
    Params:
    meanpows_norm - normalized mean powers in each frequency band
    Returns:
    X - a numpy array of the input.
    """
    # meanpows_norm=meanpows_norm.dropna(axis=0)
    params=np.unique(meanpows_norm.columns.get_level_values(1).drop(['time','mean']))#[0]    #remove '' if something breaks
    l_ecog=meanpows_norm['l_ecog'][params].reset_index(drop=True).to_numpy()
    r_ecog=meanpows_norm['r_ecog'][params].reset_index(drop=True).to_numpy()
    emg=meanpows_norm['emg','mean'].to_numpy()
    X=np.column_stack([np.concatenate([l_ecog,r_ecog],axis=1),emg])
    return X

def _classifier_has(attr):
    """Check if we can delegate a method to the underlying classifier.

    First, we check the first fitted classifier if available, otherwise we
    check the unfitted classifier.
    """
    return lambda estimator: (
        hasattr(estimator.classifier_, attr)
        if hasattr(estimator, "classifier_")
        else hasattr(estimator.classifier, attr)
    )


class InductiveClusterer(BaseEstimator):
    def __init__(self, clusterer, classifier):
        self.clusterer = clusterer
        self.classifier = classifier

    def fit(self, X, y=None):
        self.clusterer_ = clone(self.clusterer)
        self.classifier_ = clone(self.classifier)
        y = self.clusterer_.fit_predict(X)
        self.classifier_.fit(X, y)
        return self

    @available_if(_classifier_has("predict"))
    def predict(self, X):
        check_is_fitted(self)
        return self.classifier_.predict(X)

    @available_if(_classifier_has("decision_function"))
    def decision_function(self, X):
        check_is_fitted(self)
        return self.classifier_.decision_function(X)
    
def path_to_feature(sel_paths):
    """
    Function for reading data, filtering and feature extraction
    Excludes LFP data     

    Args:
        sel_paths (list): Selected paths to read

    Returns:
        features: extracted features from all paths
    """

    #output list
    allpaths_features=[]
    #reads and converts each path
    for path in sel_paths:
        if len(path)==0:
            continue
        data=read_txt([path])
        #get data in mv
        data=convert_to_mv(data)
        #convert index
        data.index=pd.to_datetime(data['time'])
        print('data read complete')
        #filtering
        filtered_data=pd.DataFrame()
        filtered_data['l_ecog']=filter_channel(data['l_ecog'], fstart=0.5, fstop=45, sr=1000, center=True, notch_50=False)
        filtered_data['r_ecog']=filter_channel(data['r_ecog'], fstart=0.5, fstop=45, sr=1000, center=True, notch_50=False)
        #LFP data not needed
        #filtered_data['lfp']=filter_channel(data['lfp'], fstart=0.5, fstop=45, sr=1000, center=True, notch_50=True)
        filtered_data['emg']=filter_channel(data['emg'], fstart=5, fstop=100, sr=1000, center=True, notch_50=True)
        filtered_data.index=data.index
        print('data filtering complete')
        #PCA of ECOG
        sel_chs=['l_ecog','r_ecog']
        pca=PCA(n_components=2)
        #select first component
        to_features=pca.fit_transform(filtered_data[sel_chs])[:,0]
        print(pca.explained_variance_ratio_)
        #spectral feature extraction
        window=1000
        feat_list = []
        #features extracted for each window
        for i in range(0,int(np.floor(len(to_features)/window))-1):
                f, pxx = signal.periodogram(to_features[i*window:(i+1)*window], fs=1000, scaling='spectrum')
                feat_list.append(pd.Series(pxx[0:46]))
        features=pd.DataFrame(feat_list)
        features.columns = features.columns.astype('str')
        features['emg_pow'], _,_=signal_to_power(filtered_data['emg'])
        #convert to relative power
        for ind in features.index:
            features.iloc[ind,:]=features.iloc[ind,:]/features.iloc[ind,:].sum()
        print('features extracted')
        #standardize
        scaler = StandardScaler()
        features[features.columns]=scaler.fit_transform(features[features.columns])
        allpaths_features.append(features)
        
    return pd.concat(allpaths_features)