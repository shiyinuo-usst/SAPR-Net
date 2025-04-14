import mne
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from mne.preprocessing import ICA
from mne.io import read_raw_edf
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
def extract_psd_features(raw, sfreq, freq_bands, n_fft):
    all_band_features = []
    for (fmin, fmax) in freq_bands:
        band_psd_means = []
        for start_idx in range(0, len(raw.times), n_fft):
            _, end_idx = start_idx, start_idx + n_fft
            if end_idx > len(raw.times):
                break
            epoch_data = raw.get_data(start=start_idx, stop=end_idx)
            psd, _ = mne.time_frequency.psd_array_welch(epoch_data, sfreq=sfreq, fmin=fmin, fmax=fmax, n_fft=n_fft)
            psd_mean = psd.mean(axis=1)
            band_psd_means.append(psd_mean)
        all_band_features.append(np.array(band_psd_means))
    combined_features = np.hstack(all_band_features)
    combined_features *= 10 ** 12
    return combined_features


subject_edf_paths  = [

]

all_subjects_data = []
freq_bands = [(4, 8),(8, 12),(13, 30),(30, 45)]
channel_names_of_interest = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
subject_label = 0
for i, edf_path in enumerate(subject_edf_paths):
    raw = read_raw_edf(edf_path, preload=True)
    raw_high = read_raw_edf(edf_path, preload=True)


    # channel_names_of_interest = ['AF3', 'F7', 'F3', 'FC5', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8', 'FC6', 'F4', 'F8', 'AF4']
    channel_names_of_interest = ['AF3', 'F7', 'T7', 'P7', 'O1', 'O2', 'P8', 'T8','F8', 'AF4']

    raw.pick_channels(channel_names_of_interest)


    # raw.plot(duration=10, scalings = {'eeg': 20e-6} , title='Original EEG Data')
    # plt.show()

    # 根据被试编号选择时间窗
    if i < 20:  # 前20个被试
        tmin, tmax =915,1065
    else:  # 后10个被试
        tmin, tmax =960,1110

    # 从原始数据中复制 AF3 通道的数据
    af3_data = raw.copy().pick_channels(['AF3'])
    # 对 AF3 通道的数据应用 0.1 到 4.0 Hz 的低通滤波器，这可能是为了提取眼电活动相关的信号
    af3_data.filter(0.1, 4.0, fir_design='firwin', picks=['AF3'])



    raw.crop(tmin, tmax).load_data()

    af3_data.crop(tmin, tmax).load_data()

    raw.set_montage('standard_1020')



    raw.set_eeg_reference(ref_channels='average', projection=True)

    # 从 AF3 通道获取数据
    EOG = af3_data.get_data()

    other_channels_data = raw.get_data(picks=channel_names_of_interest)

    new_data = np.vstack((other_channels_data, EOG))



    new_ch_names = channel_names_of_interest + ['EOG']

    # print(new_ch_names)
    new_ch_types = raw.copy().pick_channels(channel_names_of_interest).get_channel_types() + ['eog']

    new_info = mne.create_info(ch_names=new_ch_names, sfreq=raw.info['sfreq'], ch_types=new_ch_types)

    raw_with_eog = mne.io.RawArray(new_data, new_info)

    raw_with_eog.set_montage('standard_1020', on_missing='ignore')
    ica = ICA(n_components=4, random_state=97, max_iter=800)
    ica.fit(raw_with_eog)  # 拟合 ICA 到包含 EOG 的原始数据
    eog_inds, scores = ica.find_bads_eog(raw_with_eog, ch_name=['EOG'], threshold=0.9,
                                         measure='correlation')
    ica.exclude = eog_inds

    ica.apply(raw_with_eog)
    raw_without_eog = raw_with_eog.drop_channels(['EOG'])
    raw_without_eog.set_eeg_reference('average', projection=True)
    emg_inds, scores = ica.find_bads_muscle(raw_without_eog, )
    ica.exclude = emg_inds
    ica.apply(raw_without_eog)
    ica.plot_components(show=False)
    # raw_without_eog.plot()
    # plt.show()
    # raw.plot(duration=10, scalings = {'eeg': 20e-6} , title='EEG Data after EOG/EMG Artifact Removal')
    # plt.show()

    n_fft = int(2.0 * raw.info['sfreq'])  # 2秒时间窗口
    psd_features = extract_psd_features(raw_without_eog, raw.info['sfreq'], freq_bands, n_fft)

    labels = np.full((psd_features.shape[0], 1), subject_label)

    subject_data = np.hstack((psd_features, labels))

    features = subject_data[:, :-1]
    labels = subject_data[:, -1]


    subject_data_standardized = np.hstack((features, labels.reshape(-1, 1)))


    all_subjects_data.append(subject_data_standardized)

    subject_label += 1


all_subjects_array = np.vstack(all_subjects_data)

all_subjects_df = pd.DataFrame(all_subjects_array)

csv_path = ""
all_subjects_df.to_csv(csv_path, index=False, header=False)






