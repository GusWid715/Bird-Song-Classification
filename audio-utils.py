import librosa
import numpy as np

# Menentukan parameter standar untuk ekstraksi fitur.
SAMPLE_RATE = 22050
FRAME_LENGTH = 2048
HOP_LENGTH = 512
N_MFCC = 13  

def extract_features(file_path):

    try:
        # 1. Memuat file audio menggunakan librosa
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)

        # --- DOMAIN WAKTU ---
        rms = librosa.feature.rms(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        zcr = librosa.feature.zero_crossing_rate(y=audio, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        
        # Mengambil rata-rata (mean) dan standar deviasi (std) untuk merangkum fitur
        rms_mean = np.mean(rms)
        rms_std = np.std(rms)
        zcr_mean = np.mean(zcr)
        zcr_std = np.std(zcr)

        time_domain_features = np.array([rms_mean, rms_std, zcr_mean, zcr_std])

        # --- DOMAIN FREKUENSI ---
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr, frame_length=FRAME_LENGTH, hop_length=HOP_LENGTH)
        
        # Mengambil rata-rata dari fitur spektral
        spec_centroid_mean = np.mean(spec_centroid)
        spec_bandwidth_mean = np.mean(spec_bandwidth)
        
        freq_domain_features = np.array([spec_centroid_mean, spec_bandwidth_mean])

        # --- (MFCC) ---
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=N_MFCC, n_fft=FRAME_LENGTH, hop_length=HOP_LENGTH)
        
        # Mengambil rata-rata dan standar deviasi untuk setiap koefisien MFCC
        mfccs_mean = np.mean(mfccs, axis=1)
        mfccs_std = np.std(mfccs, axis=1)

        mfcc_features = np.concatenate((mfccs_mean, mfccs_std))

        # --- GABUNGKAN SEMUA FITUR MENJADI SATU VEKTOR ---
        final_features = np.concatenate((time_domain_features, freq_domain_features, mfcc_features))
        
        return final_features

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

