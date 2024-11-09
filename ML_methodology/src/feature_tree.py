from enum import Enum
import numpy as np
import scipy
import librosa
import scipy.signal as signal
from scipy.integrate import simps
from scipy.signal import butter,filtfilt
from scipy.stats import kurtosis, entropy

class IMUFeatureFamilies(str, Enum):    
    # Time Domain
    LINE_LENGTH = "ll"
    ZERO_CROSSING_RATE = "zcr"
    KURTOSIS = "kurt"
    ROOT_MEAN_SQUARED = "rms"
    CREST_FACTOR = "cf"
    APPROXIMATE_ZERO_CROSSING = "azc"

class IMU_Signal(str, Enum):
    x = "accel_x"
    y = "accel_y"
    z = "accel_z"
    Y = "gyro_y"
    P = "gyro_p"
    R = "gyro_r"
    accel = "accel_combined"
    gyro = "gyro_combined"

class AudioFeatureFamilies(str, Enum):
    # Frequency Domain
    SPECTRAL_DECREASE = "spec_decr"
    SPECTRAL_SLOPE = "spec_slope"
    SPECTRAL_ROLLOFF = "spec_roll"
    SPECTRAL_CENTROID = "spec_centr"
    SPECTRAL_SPREAD = "spec_spread"
    SPECTRAL_KURTOSIS = "spec_kurt"
    SPECTRAL_SKEW = "spec_skew"
    SPECTRAL_FLATNESS = "spec_flat"
    SPECTRAL_STD = "spec_std"
    SPECTRAL_ENTROPY = "spec_ent"
    DOMINANT_FREQUENCY = "dom_freq"
    POWER_SPECTRAL_DENSITY = "psd" #variable feature number
    
    # Mel Domain
    MEL_FREQUENCY_CEPSTRAL_COEFFICENT = "mfcc" #variable feature number
    
    # Time Domain
    ROOT_MEAN_SQUARED = "rms"
    ZERO_CROSSING_RATE = "zcr"
    CREST_FACTOR = "cf"
    ENERGY_ENVELOPE_PEAK_DETECT = "eepd" #variable feature number
    
    
def generate_feature_name_vec(n_mfcc, psd_bands, eepd_bands):
    feat_name_vec = []
    feat_type_counts = []
    for feat_fam in AudioFeatureFamilies:
        if feat_fam == AudioFeatureFamilies.POWER_SPECTRAL_DENSITY:
            for lf,hf in psd_bands:
                feat_name_vec.append(feat_fam.value + "_" + str(lf) + "_" + str(hf))
            feat_type_counts.append(len(psd_bands))
        
        elif feat_fam == AudioFeatureFamilies.MEL_FREQUENCY_CEPSTRAL_COEFFICENT:
            for i in range(n_mfcc):
                feat_name_vec.append(feat_fam.value + "_mean_" + str(i))
            for i in range(n_mfcc):
                feat_name_vec.append(feat_fam.value + "_std_" + str(i))
            for i in range(n_mfcc):
                feat_name_vec.append(feat_fam.value + "_max_" + str(i))
            for i in range(n_mfcc):
                feat_name_vec.append(feat_fam.value + "_entropy_" + str(i))
            feat_type_counts.append(n_mfcc*4)
        elif feat_fam == AudioFeatureFamilies.ENERGY_ENVELOPE_PEAK_DETECT:
            for f in eepd_bands:
                feat_name_vec.append(feat_fam.value + "_" + str(f))
            feat_type_counts.append(len(eepd_bands))
        else:
            feat_name_vec.append(feat_fam.value)
            feat_type_counts.append(1)

    return np.array(feat_name_vec), np.array(feat_type_counts)

def generate_feature_name_vec_imu(DP_epsilon):
    feat_name_vec = []
    feat_type_counts = []
    for signal_type in IMU_Signal:
        for feat_fam in IMUFeatureFamilies:
            if feat_fam == IMUFeatureFamilies.APPROXIMATE_ZERO_CROSSING:
                for eps_v in DP_epsilon:
                    feat_name_vec.append(signal_type.value + "_" + feat_fam.value + "_" + str(eps_v))
                feat_type_counts.append(len(DP_epsilon))
            else:
                feat_name_vec.append( signal_type.value + "_" + feat_fam.value)
                feat_type_counts.append(1)
    return np.array(feat_name_vec), np.array(feat_type_counts)

class AudioFeatures:
    
    def __init__(self, feat_select_vector, feat_type_counts, feat_names, n_mfcc, psd_bands, eepd_bands, compute_mfcc):
        self.select_vec = feat_select_vector
        self.feat_counts = feat_type_counts
        self.names = feat_names
        self.n_mfcc = n_mfcc
        self.psd_bands = np.array(psd_bands)
        self.eepd_bands = np.array(eepd_bands)
        self.compute_mfcc = compute_mfcc
        self.parse_feat_select_vec()
    
    def parse_feat_select_vec(self):
        #Index of each feature family
        count = 0
        for i, fam in enumerate(AudioFeatureFamilies):
            setattr(self,fam.value + "_idx",[count, count + self.feat_counts[i]])
            count += self.feat_counts[i]
        
        # Determine if each feature family is required to be computed
        for fam in AudioFeatureFamilies:
            fam_idx = getattr(self,fam.value + "_idx")
            setattr(self, "requires_" + fam.value, np.any(self.select_vec[fam_idx[0]:fam_idx[1]]))
        
        # Adjust for features that depend on each other
        self.requires_spec_centr_all = (self.requires_spec_centr | self.requires_spec_spread | self.requires_spec_kurt | self.requires_spec_skew)
        self.requires_spec_spread_all = (self.requires_spec_spread | self.requires_spec_kurt | self.requires_spec_skew)
        self.requires_rms_all = self.requires_rms | self.requires_cf
        
        # Determine if each helper function is needed
        self.requires_fft = np.any(self.select_vec[self.spec_decr_idx[0]:self.spec_skew_idx[1]])
        self.requires_periodogram = np.any(self.select_vec[self.psd_idx[0]:self.psd_idx[1]]) | self.requires_spec_flat | self.requires_spec_std | self.requires_dom_freq
        self.requires_mean =  self.requires_rms | self.requires_zcr | self.requires_cf

    #### FEATURE COMPUTATION ####
    def compute_features(self,x, fs, welch=True, psd_nperseg=900, psd_noverlap=600):
        """Computes the features selected by the inputted feature select vector on audio signal x with sampling frequency fs"""
        
        feature_array = []
        feature_names = []
        
        # FFT-based features
        if self.requires_fft:
            magnitudes = self.real_fft_mag(x)
            frequencies = self.real_fft_freqs(x,fs)
            sum_mags = self.sum_fft_magnitudes(magnitudes)
            
            #Spectral decrease
            if self.requires_spec_decr:
                feature_array.append(self.spectral_decrease(magnitudes,frequencies,sum_mags))
                feature_names.append(self.names[self.spec_decr_idx[0]:self.spec_decr_idx[1]].item())
            
            #Spectral slope
            if self.requires_spec_slope:
                feature_array.append(self.spectral_slope(magnitudes,frequencies,sum_mags))
                feature_names.append(self.names[self.spec_slope_idx[0]:self.spec_slope_idx[1]].item())
            
            #Spectral rolloff
            if self.requires_spec_roll:
                feature_array.append(self.spectral_rolloff(magnitudes,frequencies,sum_mags))
                feature_names.append(self.names[self.spec_roll_idx[0]:self.spec_roll_idx[1]].item())
            
            #Spectral centroid
            if self.requires_spec_centr_all:
                centroid = self.spectral_centroid(magnitudes,frequencies,sum_mags)
                if self.requires_spec_centr:
                    feature_array.append(centroid)
                    feature_names.append(self.names[self.spec_centr_idx[0]:self.spec_centr_idx[1]].item())
                
                #Spectral spread
                if self.requires_spec_spread_all:
                    spread = self.spectral_spread(magnitudes,frequencies,sum_mags,centroid)
                    if self.requires_spec_spread:
                        feature_array.append(spread)
                        feature_names.append(self.names[self.spec_spread_idx[0]:self.spec_spread_idx[1]].item())
                    
                    #Spectral kurtosis
                    if self.requires_spec_kurt:
                        feature_array.append(self.spectral_kurtosis(magnitudes,frequencies,sum_mags,centroid,spread))
                        feature_names.append(self.names[self.spec_kurt_idx[0]:self.spec_kurt_idx[1]].item())
                    
                    #Spectral skew
                    if self.requires_spec_skew:
                        feature_array.append(self.spectral_skewness(magnitudes,frequencies,sum_mags,centroid,spread))
                        feature_names.append(self.names[self.spec_skew_idx[0]:self.spec_skew_idx[1]].item())
        
        # periodogram-based features
        if self.requires_periodogram:
            freqs, psd = self.periodogram(x, fs, welch=welch, nperseg=psd_nperseg, noverlap=psd_noverlap)
            
            #Spectral flatness
            if self.requires_spec_flat:
                feature_array.append(self.spectral_flatness(psd))
                feature_names.append(self.names[self.spec_flat_idx[0]:self.spec_flat_idx[1]].item())
            
            #Spectral standard deviation
            if self.requires_spec_std:
                feature_array.append(self.spectral_std(psd))
                feature_names.append(self.names[self.spec_std_idx[0]:self.spec_std_idx[1]].item())
                
            #Spectral entropy
            if self.requires_spec_ent:
                feature_array.append(self.spectral_entropy(psd))
                feature_names.append(self.names[self.spec_ent_idx[0]:self.spec_ent_idx[1]].item())
            
            #Dominant frequency
            if self.requires_dom_freq:
                feature_array.append(self.dominant_frequency(psd,freqs))
                feature_names.append(self.names[self.dom_freq_idx[0]:self.dom_freq_idx[1]].item())
            
            #PSD
            if self.requires_psd:
                out = self.normalized_bandpower(psd,freqs)
                for feat in out:
                    feature_array.append(feat)
                psd_names = self.names[self.psd_idx[0]:self.psd_idx[1]][self.select_vec[self.psd_idx[0]:self.psd_idx[1]]]
                for n in psd_names:
                    feature_names.append(n)
        # mfccs
        if self.requires_mfcc:
            out = self.mfcc(x,fs)
            for mfcc_feat in out:
                feature_array.append(mfcc_feat)
            mfcc_names = self.names[self.mfcc_idx[0]:self.mfcc_idx[1]][self.select_vec[self.mfcc_idx[0]:self.mfcc_idx[1]]]
            for n in mfcc_names:
                feature_names.append(n)
                
        #Mean-based features
        if self.requires_mean:
            x_zero_mean = self.sub_mean(x)
            
            #Zero crossing rate
            if self.requires_zcr:
                feature_array.append(self.zcr(x_zero_mean))
                feature_names.append(self.names[self.zcr_idx[0]:self.zcr_idx[1]].item())
            
            if self.requires_rms_all:
                rms_power = self.rms(x_zero_mean)
                
                #RMS power
                if self.requires_rms:
                    feature_array.append(rms_power)
                    feature_names.append(self.names[self.rms_idx[0]:self.rms_idx[1]].item())
                
                #Crest factor
                if self.requires_cf:
                    feature_array.append(self.crest_factor(x_zero_mean,rms_power))
                    feature_names.append(self.names[self.cf_idx[0]:self.cf_idx[1]].item())
        
        #EEPD
        if self.requires_eepd:
            out = self.eepd(x,fs)
            for eepd_feat in out:
                feature_array.append(eepd_feat)
            eepd_names = self.names[self.eepd_idx[0]:self.eepd_idx[1]][self.select_vec[self.eepd_idx[0]:self.eepd_idx[1]]]
            for n in eepd_names:
                feature_names.append(n)
                    
                        
        return np.array(feature_array), np.array(feature_names)

    
    #### FEATURE EXTRACTION FUNCTIONS #######
    #### FREQUENCY DOMAIN ####
    
    # FFT-based helper functions #
    def real_fft_mag(self, x):
        """Magnitudes of the FFT of a real signal x (i.e. only positive frequencies)"""
        magnitudes = np.abs(np.fft.rfft(x))
        return magnitudes
    
    def real_fft_freqs(self,x,fs):
        """Returns the frequencies corresponding to each FFT magnitude of signal x with sampling rate fs"""
        length = len(x)
        freqs = np.fft.rfftfreq(length, d=1/fs)
        return freqs
    
    def sum_fft_magnitudes(self,mags):
        """Sum of all magnitudes of FFT coefficients"""
        return  np.sum(mags) + 1e-17
    
    # FFT-based features #
    def spectral_centroid(self, mags, freqs, sum_mag):
        """"Frequency at which the center of mass of the spectrum is located.
        Uses FFT magnitudes, their corresponding frequencies, and the sum of all FFT magnitudes"""
        return np.sum(mags*freqs) / sum_mag
    
    def spectral_rolloff(self,mags,freqs, sum_mag):
        """Frequency below which 95% of the spectral energy is located"""
        cumsum_mag = np.cumsum(mags)
        return np.min(freqs[np.where(cumsum_mag >= 0.95*sum_mag)[0]]) 
    
    def spectral_spread(self,mags,freqs,sum_mag, centroid):
        """Weighted standard deviation of frequencies with respect to the FFT magnitudes.
        Uses the previously-computed spectral centroid."""
        return np.sqrt(np.sum(((freqs-centroid)**2)*mags) / sum_mag)
    
    def spectral_skewness(self,mags,freqs,sum_mag, centroid, spread):
        """Measure the symmetry of the spectrum around its arithmetic mean value.
        Uses the previously-computed spectral centroid and spread."""
        return np.sum(((freqs-centroid)**3)*mags) / ((spread**3)*sum_mag)
    
    def spectral_kurtosis(self,mags,freqs,sum_mag,centroid,spread):
        """Describes the flatness of the spectrum around its mean value.
        Uses the previously-computed spectral centroid and spread."""
        return np.sum(((freqs-centroid)**4)*mags) / ((spread**4)*sum_mag)
    
    def spectral_slope(self,mags,freqs, sum_mag):
        """Measure of slope of the spectrum of the signal"""
        n_freq = len(freqs)
        mean_spec = sum_mag/n_freq
        mean_freq = sum(freqs)/n_freq
        return (freqs - mean_freq).dot(mags - mean_spec) / (np.sum((freqs-mean_freq)**2))
    
    def spectral_decrease(self,mags,freqs,sum_mag):
        """Average spectral-slope of the rate-map representation"""
        dc_coeff = mags[0]
        return np.sum((mags - dc_coeff)/(np.arange(len(mags))+1)) / sum_mag
        
    # Periodogram-based helper functions #
    
    def periodogram(self, x, fs, welch=True, nperseg=900, noverlap=600):
        """Compute the periodogram of the signal (i.e. the signal power in each frequency bin).
        Use either Welch's method (more computationally expensive but more stable features) or a classic periodogram"""
        if welch:
            nperseg = min(nperseg,len(x))
            noverlap=min(noverlap,int(nperseg/2))
            freqs, psd = signal.welch(x, fs, nperseg=nperseg, noverlap=noverlap)
        else: 
            freqs, psd = signal.periodogram(x, fs)
        return freqs, psd
    
    # Periodogram-based features #
    def spectral_flatness(self, psd):
        """Measure of the uniformity in the frequency distribution of the power spectrum"""
        psd_len = len(psd)
        gmean = np.exp((1/psd_len)*np.sum(np.log(psd + 1e-17)))
        amean = (1/psd_len)*np.sum(psd)
        return gmean/(amean + 1e-17)
    
    def spectral_std(self, psd):
        """Standard deviation of the PSD"""
        return np.std(psd)
    
    def dominant_frequency(self,psd,freqs):
        "Value of the frequency bin at which the maximum power of the signal is found"
        return freqs[np.argmax(psd)]
    
    def normalized_bandpower(self,psd,freqs):
        """Integral of the PSD in given frequency bands normalized to the total signal power"""
        dx_freq = freqs[1]-freqs[0]
        total_power = simps(psd, dx=dx_freq)
        #Use only the frequency bands that are required by the feature select vector
        bands_to_compute = self.psd_bands[self.select_vec[self.psd_idx[0]:self.psd_idx[1]]]
        out = np.zeros(len(bands_to_compute))
        for i, (lf, hf) in enumerate(bands_to_compute):
            idx_band = np.logical_and(freqs >= lf, freqs <= hf)
            band_power = simps(psd[idx_band], dx=dx_freq)
            out[i] = band_power/total_power
        return out

    def _xlogx(self, x, base=2):
        """Returns x log_b x if x is positive, 0 if x == 0, and np.nan
        otherwise. This handles the case when the power spectrum density
        takes any zero value. From antropy package.
        """
        x = np.asarray(x)
        xlogx = np.zeros(x.shape)
        xlogx[x < 0] = np.nan
        valid = x > 0
        xlogx[valid] = x[valid] * np.log(x[valid]) / np.log(base)
        return xlogx
    
    def spectral_entropy(self, _psd):
        """Entropy of the power spectrum
        """
        psd_norm = _psd / _psd.sum(axis=-1, keepdims=True)
        se = -self._xlogx(psd_norm).sum(axis=-1)
        return se

    # MFCCs #
    def mfcc(self, x, fs):
        """Compute the mean and standard deviation MFCCs or mel spectral components of the signal over time"""
        if self.compute_mfcc:
            mfcc = librosa.feature.mfcc(y = x, sr = fs, n_mfcc = self.n_mfcc)
            mfcc_entropy = scipy.stats.entropy(mfcc, axis=1)
            mfcc_entropy[np.isinf(mfcc_entropy)] = 0
        else:
            Sxx = librosa.feature.melspectrogram(y=x, sr=fs, n_mels=self.n_mfcc)
            mfcc_entropy = scipy.stats.entropy(Sxx, axis=1)
            mfcc = librosa.power_to_db(Sxx)
        mfcc_mean = mfcc.mean(axis=1)
        mfcc_std = mfcc.std(axis=1)
        mfcc_max = mfcc.max(axis=1)
        all_features = np.concatenate((mfcc_mean, mfcc_std, mfcc_max, mfcc_entropy))
        return all_features[self.select_vec[self.mfcc_idx[0]:self.mfcc_idx[1]]]
    
    #### TIME DOMAIN ####
    
    def sub_mean(self, x):
        """Subtract the mean of a signal"""
        return x - np.mean(x)
    
    def rms(self,x):
        """Compute the root mean squared power of the signal"""
        return np.sqrt(np.mean(np.square(x)))
    
    def zcr(self,x):
        """Compute the zero-crossing rate of the signal"""
        return (np.sum(np.multiply(x[0:-1],x[1:])<0)/(len(x)-1))
    
    def crest_factor(self,x,rmsp):
        """Crest factor = ratio of the peak of the signal to its RMS power"""
        peak = np.max(x)
        return peak/rmsp

    def eepd(self,x,fs):
        """Energy envelope peak detection: number of peaks in a given energy envelope of a bandpassed signal"""
        fNyq = fs/2
        bands_to_compute = self.eepd_bands[self.select_vec[self.eepd_idx[0]:self.eepd_idx[1]]]
        out = np.zeros(len(bands_to_compute))
        eepd_bandwidth = self.eepd_bands[1]-self.eepd_bands[0]
        for i, fcl in enumerate(bands_to_compute):
            fc = [fcl/fNyq, (fcl+eepd_bandwidth)/fNyq]
            b, a = butter(1, fc, btype='bandpass')
            bpFilt = filtfilt(b, a, x)
            b,a = butter(2, 10/fNyq, btype='lowpass')
            eed = filtfilt(b, a, bpFilt**2)
            eed = eed/np.max(eed+1e-17)
            peaks,_ = signal.find_peaks(eed)
            out[i] = len(peaks)
        return out
class IMUFeatures:
    
    def __init__(self, feat_select_vector, feat_type_counts, feat_names, DP_epsilon=[0.5]):
        self.select_vec = feat_select_vector
        self.feat_counts = feat_type_counts
        self.names = feat_names
        self.DP_epsilon = DP_epsilon
        self.parse_feat_select_vec()
        self.DP_epsilon = DP_epsilon
    
    def parse_feat_select_vec(self):
        #Index of each feature family
        count = 0
        for signal_type in IMU_Signal:
            for i, feat_fam in enumerate(IMUFeatureFamilies):
                setattr(self, signal_type.value + "_" + feat_fam.value + "_idx",[count, count + self.feat_counts[i]])
                count += self.feat_counts[i]
        
        # Determine if each feature family is required to be computed
        for signal_type in IMU_Signal:
            for feat_fam in IMUFeatureFamilies:
                fam_idx = getattr(self, signal_type.value + "_" + feat_fam.value + "_idx")

                # Determine if AZC feature is required for each value of tolerance DP_epsilon
                if feat_fam == "azc":
                    for i, eps in enumerate(self.DP_epsilon):
                        setattr(self, "requires_" + signal_type.value + "_" + feat_fam.value + "_" + str(eps), np.any(self.select_vec[fam_idx[0]+i]))
                else:
                    setattr(self, "requires_" + signal_type.value + "_" + feat_fam.value, np.any(self.select_vec[fam_idx[0]:fam_idx[1]]))
        
        #Check if each signal is required
        n_feats  = len(IMUFeatureFamilies)
        start_feature = ""
        end_feature = ""
        for i, ft in enumerate(IMUFeatureFamilies):
            if i == 0:
                start_feature = ft.value
            elif i == n_feats - 1:
                end_feature = ft.value
        for signal_type in IMU_Signal:
            fam_idx_start = getattr(self,signal_type.value + "_" + start_feature + "_idx")
            fam_idx_end = getattr(self,signal_type.value + "_" + end_feature + "_idx")
            setattr(self,"requires_" + signal_type.value,  np.any(self.select_vec[fam_idx_start[0]:fam_idx_end[1]]))
        
        for signal_type in IMU_Signal:
            # Adjust for features that depend on each other
            setattr(self,"requires_" + signal_type.value + "_rms_all", getattr(self, "requires_" + signal_type.value + "_rms") | getattr(self, "requires_" + signal_type.value + "_cf"))
        

    #### FEATURE COMPUTATION ####
    def compute_features(self,x,fs,signal_type):
        """Computes the features selected by the inputted feature select vector on audio signal x with sampling frequency fs"""
        
        feature_array = []
        feature_names = []
        
        signal_needed = getattr(self,"requires_"+signal_type.value)
        if signal_needed:
            #Line length
            if getattr(self, "requires_" + signal_type.value + "_ll"):
                idx = getattr(self, signal_type.value + "_ll_idx")
                feature_array.append(self.line_length(x))
                feature_names.append(self.names[idx[0]:idx[1]].item())
                
            #Zero-crossing rate
            if getattr(self, "requires_" + signal_type.value + "_zcr"):
                idx = getattr(self, signal_type.value + "_zcr_idx")
                feature_array.append(self.zcr(x))
                feature_names.append(self.names[idx[0]:idx[1]].item())

            #Kurtosis
            if getattr(self, "requires_" + signal_type.value + "_kurt"):
                idx = getattr(self, signal_type.value + "_kurt_idx")
                feature_array.append(self.sig_kurtosis(x))
                feature_names.append(self.names[idx[0]:idx[1]].item())


            #RMS-based
            if getattr(self, "requires_" + signal_type.value + "_rms_all"):
                #RMS
                idx = getattr(self, signal_type.value + "_rms_idx")
                rmsp = self.rms(x)
                if getattr(self, "requires_" + signal_type.value + "_rms"):
                    feature_array.append(rmsp)
                    feature_names.append(self.names[idx[0]:idx[1]].item())
                    
                #Crest factor
                if getattr(self, "requires_" + signal_type.value + "_cf"):
                    idx = getattr(self, signal_type.value + "_cf_idx")
                    feature_array.append(self.crest_factor(x, rmsp))
                    feature_names.append(self.names[idx[0]:idx[1]].item())
                    
            # #Approximate Zero Crossing
            for i, eps in enumerate(self.DP_epsilon):
                if getattr(self, "requires_" + signal_type.value + "_azc" + "_" + str(eps)):
                    idx = getattr(self, signal_type.value + "_azc_idx")
                    azc = self.azc_computation(x, epsilon=eps)
                    feature_array.append(azc)
                    feature_names.append(self.names[idx[0] + i])

        return np.array(feature_array), np.array(feature_names)

    
    #### FEATURE EXTRACTION FUNCTIONS #######
    #### TIME DOMAIN ####
    
    def get_mean(self,x):
        return np.mean(x)
    
    def sub_mean(self, x, mean):
        """Subtract the mean of a signal"""
        return x - mean
    
    def rms(self,x):
        """Compute the root mean squared power of the signal"""
        return np.sqrt(np.mean(np.square(x)))
    
    def zcr(self,x):
        """Compute the zero-crossing rate of the signal"""
        return (np.sum(np.multiply(x[0:-1],x[1:])<0)/(len(x)-1))
    
    def crest_factor(self,x,rmsp):
        """Crest factor = ratio of the peak of the signal to its RMS power"""
        peak = np.max(x)
        return peak/rmsp
    
    def line_length(self, x):
        """Line length of the signal segment: mean absolute difference between consecutive points"""
        return np.mean(np.abs(np.diff(x)))
    
    def sig_kurtosis(self,x):
        """Signal kurtosis = 'tailedness' of the signal value distribution"""
        return kurtosis(x)

    def __polygonal_approx(self, arr, epsilon=0.5):
        """
        Performs an optimized version of the Ramer-Douglas-Peucker algorithm assuming as an input
        an array of single values, considered consecutive points, and **taking into account only the
        vertical distances**.
        """
        def max_vdist(arr, first, last):
            """
            Obtains the distance and the index of the point in *arr* with maximum vertical distance to
            the line delimited by the first and last indices. Returns a tuple (dist, index).
            """
            if first == last:
                return (0.0, first)
            frg = arr[first:last+1]
            leng = last-first+1
            dist = np.abs(frg - np.interp(np.arange(leng),[0, leng-1], [frg[0], frg[-1]]))
            idx = np.argmax(dist)
            return (dist[idx], first+idx)

        if epsilon <= 0.0:
            raise ValueError('Epsilon must be > 0.0')
        if len(arr) < 3:
            return arr
        result = set()
        stack = [(0, len(arr) - 1)]
        while stack:
            first, last = stack.pop()
            max_dist, idx = max_vdist(arr, first, last)
            if max_dist > epsilon:
                stack.extend([(first, idx),(idx, last)])
            else:
                result.update((first, last))
        return np.array(sorted(result))
        

    def azc_computation(self, x, epsilon=0.5):
        """ Returns the number of peaks of the signal x after having approximated it with the Duglas-Peucker algorithm """
        timestamps = np.arange(0, len(x))

        idxs = self.__polygonal_approx(x, epsilon=epsilon)
        x_approx = x[idxs]
        times_approx = timestamps[idxs]

        # Apply discrete time differentiation
        diff_x = []
        for i in range(len(x_approx) - 1):
            diff_x.append((x_approx[i+1] - x_approx[i]) / (times_approx[i+1] - times_approx[i]))
        
        if len(diff_x) > 1:
            return np.sum(np.multiply(diff_x[0:-1],diff_x[1:])<0)
        else:
            return 0
