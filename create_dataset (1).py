import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

def get_file_path(folder_path, keyword, exclude_keyword=None):
    search_pattern = os.path.join(folder_path, f"*{keyword}*.txt")
    files = glob.glob(search_pattern)
    if not files: return None
    if exclude_keyword:
        filtered_files = [f for f in files if exclude_keyword not in f]
        if filtered_files: return filtered_files[0]
    return files[0]

def parse_continuous_file(filepath):
    if filepath is None: return pd.DataFrame()
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "Data:":
            data_start = i + 1
            break
    records = []
    for line in lines[data_start:]:
        if ';' in line:
            parts = line.strip().split(';')
            time_str = parts[0].strip().replace(',', '.')
            try:
                val = float(parts[1].strip()) 
                records.append({'Timestamp': time_str, 'Value': val})
            except ValueError: continue
    df = pd.DataFrame(records)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        df.set_index('Timestamp', inplace=True)
    return df

def parse_events(filepath):
    events = []
    if filepath is None: return events
    with open(filepath, 'r') as file:
        lines = file.readlines()
    data_start = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            data_start = i + 1
            break
    for line in lines[data_start:]:
        if ';' in line:
            parts = line.strip().split(';')
            time_range = parts[0].strip()
            if '-' not in time_range: continue
            start_str, end_time_str = time_range.split('-')
            start_str = start_str.replace(',', '.')
            end_time_str = end_time_str.replace(',', '.')
            event_type = parts[2].strip().lower()
            if 'apnea' in event_type or 'hypopnea' in event_type:
                start_dt = pd.to_datetime(start_str, format='%d.%m.%Y %H:%M:%S.%f')
                end_dt_str = start_dt.strftime('%d.%m.%Y') + ' ' + end_time_str
                end_dt = pd.to_datetime(end_dt_str, format='%d.%m.%Y %H:%M:%S.%f')
                if end_dt < start_dt: end_dt += pd.Timedelta(days=1)
                events.append({'start': start_dt, 'end': end_dt})
    return events

def apply_bandpass_filter(data, lowcut=0.17, highcut=0.4, fs=4.0, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data, padlen=min(len(data)-1, 15)) 
    return y

def check_overlap(window_start, window_end, events):
    overlap_duration = 0
    for ev in events:
        latest_start = max(window_start, ev['start'])
        earliest_end = min(window_end, ev['end'])
        delta = (earliest_end - latest_start).total_seconds()
        if delta > 0: overlap_duration += delta
    return 1 if overlap_duration >= 15 else 0

def process_participant(folder_path):
    participant_id = os.path.basename(os.path.normpath(folder_path))
    print(f"Processing {participant_id}...")
    
    path_flow = get_file_path(folder_path, "Flow", exclude_keyword="Events")
    path_thorac = get_file_path(folder_path, "Thorac")
    path_spo2 = get_file_path(folder_path, "SPO2")
    path_events = get_file_path(folder_path, "Events")
    
    if not path_flow or not path_thorac or not path_spo2:
        return pd.DataFrame()
        
    df_flow = parse_continuous_file(path_flow)
    df_thorac = parse_continuous_file(path_thorac)
    df_spo2 = parse_continuous_file(path_spo2)
    events = parse_events(path_events)
    
    if df_flow.empty or df_thorac.empty or df_spo2.empty:
        return pd.DataFrame()
    
    df_flow = df_flow.resample('250ms').mean().ffill().bfill()
    df_thorac = df_thorac.resample('250ms').mean().ffill().bfill()
    df_spo2 = df_spo2.resample('250ms').mean().ffill().bfill()
    
    combined_df = df_flow.join([df_thorac.rename(columns={'Value':'Thorac'}), 
                                df_spo2.rename(columns={'Value':'SpO2'})], how='inner')
    
    if combined_df.empty: return pd.DataFrame()
        
    combined_df.rename(columns={'Value':'Flow'}, inplace=True)
    combined_df['Flow_Filtered'] = apply_bandpass_filter(combined_df['Flow'].values)
    combined_df['Thorac_Filtered'] = apply_bandpass_filter(combined_df['Thorac'].values)
    
    window_size = pd.Timedelta(seconds=30)
    step_size = pd.Timedelta(seconds=15)
    start_time = combined_df.index[0]
    end_time = combined_df.index[-1]
    
    dataset_rows = []
    current_time = start_time
    
    while current_time + window_size <= end_time:
        window_end = current_time + window_size
        label = check_overlap(current_time, window_end, events)
        window_data = combined_df.loc[current_time:window_end - pd.Timedelta(milliseconds=1)]
        
        if len(window_data) == 120:
            signal_matrix = window_data[['Flow_Filtered', 'Thorac_Filtered', 'SpO2']].values
            row = {'Participant': participant_id, 'Start_Time': current_time, 'Label': label, 'Signal_Matrix': signal_matrix}
            dataset_rows.append(row)
        current_time += step_size
        
    return pd.DataFrame(dataset_rows)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-in_dir', type=str, required=True)
    parser.add_argument('-out_dir', type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_data = []
    
    participant_folders = [f.path for f in os.scandir(args.in_dir) if f.is_dir() and f.name.startswith('AP')]
    participant_folders.sort()
    
    for folder in participant_folders:
        df = process_participant(folder)
        if not df.empty: all_data.append(df)
        
    if all_data:
        final_dataset = pd.concat(all_data, ignore_index=True)
        save_path = os.path.join(args.out_dir, 'breathing_dataset.pkl')
        final_dataset.to_pickle(save_path)
        print(f"\n SUCCESS! Master Pickle Dataset saved to {save_path}")

if __name__ == "__main__":
    main()
