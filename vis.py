import os
import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def get_file_path(folder_path, keyword, exclude_keyword=None):

    search_pattern = os.path.join(folder_path, f"*{keyword}*.txt")
    files = glob.glob(search_pattern)
    if not files: return None
    if exclude_keyword:
        filtered_files = [f for f in files if exclude_keyword not in f]
        if filtered_files: return filtered_files[0]
    return files[0]

def parse_continuous_file(filepath):
    """Reads physiological data files."""
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
            except ValueError:
                continue

    df = pd.DataFrame(records)
    if not df.empty:
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%d.%m.%Y %H:%M:%S.%f', errors='coerce')
        df.dropna(subset=['Timestamp'], inplace=True)
        df.set_index('Timestamp', inplace=True)
    return df

def parse_events(filepath):
    """Reads the Flow Events file to get start and end times of Apnea."""
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
                if end_dt < start_dt:
                    end_dt += pd.Timedelta(days=1)
                
                events.append({'start': start_dt, 'end': end_dt, 'type': event_type})
    return events

def main():
    # Set up command line arguments (Assignment requirement)
    parser = argparse.ArgumentParser(description='Visualize Sleep Data')
    parser.add_argument('-name', type=str, required=True, help='Path to participant folder (e.g., Data/AP01)')
    args = parser.parse_args()

    folder_path = args.name
    participant_id = os.path.basename(os.path.normpath(folder_path))
    
    print(f"Loading data for {participant_id}...")
    
    # 1. Find Files
    path_flow = get_file_path(folder_path, "Flow", exclude_keyword="Events")
    path_thorac = get_file_path(folder_path, "Thorac")
    path_spo2 = get_file_path(folder_path, "SPO2")
    path_events = get_file_path(folder_path, "Events")
    
    # 2. Parse Data
    df_flow = parse_continuous_file(path_flow)
    df_thorac = parse_continuous_file(path_thorac)
    df_spo2 = parse_continuous_file(path_spo2)
    events = parse_events(path_events)
    
    # To prevent the PDF from being massively huge (millions of points), we resample to 1Hz
    if not df_flow.empty: df_flow = df_flow.resample('1S').mean()
    if not df_thorac.empty: df_thorac = df_thorac.resample('1S').mean()
    if not df_spo2.empty: df_spo2 = df_spo2.resample('1S').mean()

    print("Plotting the signals...")
    # 3. Create the Plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f'8-Hour Sleep Profile & Apnea Events: {participant_id}', fontsize=16)

    # Plot Nasal Flow
    if not df_flow.empty:
        ax1.plot(df_flow.index, df_flow['Value'], color='blue', linewidth=0.5)
    ax1.set_ylabel('Nasal Flow\n(L/min)')
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot Thoracic Movement
    if not df_thorac.empty:
        ax2.plot(df_thorac.index, df_thorac['Value'], color='green', linewidth=0.5)
    ax2.set_ylabel('Resp. Amplitude\n(Thoracic)')
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Plot SpO2
    if not df_spo2.empty:
        ax3.plot(df_spo2.index, df_spo2['Value'], color='red', linewidth=1)
    ax3.set_ylabel('SpO2\n(%)')
    ax3.set_xlabel('Time')
    ax3.grid(True, linestyle='--', alpha=0.5)

    

    # 4. Overlay Events (Red shaded regions)
    for ax in [ax1, ax2, ax3]:
        for event in events:
            ax.axvspan(event['start'], event['end'], color='red', alpha=0.3)

    # Format x-axis to show time properly
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.tight_layout()

    # 5. Save to PDF
    vis_dir = "Visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    save_path = os.path.join(vis_dir, f"{participant_id}_visualization.pdf")
    plt.savefig(save_path, format='pdf', dpi=300)
    plt.close()
    
    print(f"Success! Visualization saved at: {save_path}")

if __name__ == "__main__":
    main()
