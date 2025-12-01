import pandas as pd
import numpy as np
import os
import h5py
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# 常量定义
TOTAL_ROWS = 1800
TOTAL_COLS = 3600
LAT_START = 89.95
LON_START = -179.95
RESOLUTION = 0.1

TARGET_VARIABLES = [
    'VOD_Ku_Hpol_Asc', 'VOD_Ku_Vpol_Asc', 'VOD_X_Hpol_Asc', 'VOD_X_Vpol_Asc',
    'VOD_C_Hpol_Asc', 'VOD_C_Vpol_Asc', 'LAI', 'Hveg', 'Grass_man', 'Grass_nat',
    'Shrub_bd', 'Shrub_be', 'Shrub_nd', 'Shrub_ne', 'Tree_bd', 'Tree_be', 'Tree_nd', 'Tree_ne'
]

class XLSXProcessor:
    def __init__(self, vod_path=None, pft_path=None, lai_path=None, hveg_path=None):
        self.vod_path = vod_path or r"E:\data\VOD\mat\kuxcVOD\ASC\\"
        self.pft_path = pft_path or r"E:\data\ESACCI PFT\Resample\Data\\"
        self.lai_path = lai_path or r"E:\data\GLASS LAI\mat\0.1Deg\Dataset\\"
        self.hveg_path = hveg_path or r"E:\data\CanopyHeight\CH.mat"
        self.vod_cache = {}
        self.pft_cache = {}
        self.lai_cache = {}
        self.hveg_data = None

    @staticmethod
    def calculate_grid_indices(lon, lat):
        row = int(round((LAT_START - lat) / RESOLUTION))
        col = int(round((lon - LON_START) / RESOLUTION))
        row = max(0, min(row, TOTAL_ROWS - 1))
        col = max(0, min(col, TOTAL_COLS - 1))
        return row, col

    @staticmethod
    def safe_date_to_str(date_val):
        if pd.isna(date_val):
            return ""
        if isinstance(date_val, datetime):
            return date_val.strftime('%Y%m%d')
        elif isinstance(date_val, np.datetime64):
            return pd.to_datetime(date_val).strftime('%Y%m%d')
        elif isinstance(date_val, (int, float)):
            date_str = str(int(date_val))
            return date_str[:8] if len(date_str) > 8 else date_str.zfill(8)
        else:
            date_str = str(date_val).replace('-', '').replace('/', '').replace(' ', '')
            return date_str[:8] if len(date_str) > 8 else date_str.zfill(8)

    @staticmethod
    def read_mat_file(file_path, variable_names, silent=False):
        try:
            with h5py.File(file_path, 'r') as f:
                data = {}
                for var in variable_names:
                    if var in f:
                        dataset = f[var]
                        if isinstance(dataset, h5py.Reference):
                            dataset = f[dataset]
                        if len(dataset.shape) == 2:
                            matrix = dataset[()]
                            if matrix.shape == (TOTAL_ROWS, TOTAL_COLS):
                                data[var] = matrix
                            elif matrix.shape == (TOTAL_COLS, TOTAL_ROWS):
                                data[var] = matrix.T
                            else:
                                try:
                                    data[var] = matrix.reshape(TOTAL_ROWS, TOTAL_COLS)
                                except:
                                    data[var] = np.full((TOTAL_ROWS, TOTAL_COLS), np.nan)
                        else:
                            data[var] = np.full((TOTAL_ROWS, TOTAL_COLS), np.nan)
                return data
        except Exception as e:
            if not silent:
                print(f"警告: 读取文件 {file_path} 时出错: {str(e)}")
            return None

    @staticmethod
    def calculate_lai_weight(date_str):
        if len(date_str) != 8 or not date_str.isdigit():
            return None, None, 0.0
        try:
            year = int(date_str[:4])
            month = int(date_str[4:6])
            day = int(date_str[6:8])
            current_date = datetime(year, month, day)
        except:
            return None, None, 0.0
        if day < 15:
            prev_month = month - 1
            prev_year = year
            if prev_month == 0:
                prev_month = 12
                prev_year = year - 1
            prev_month_mid = datetime(prev_year, prev_month, 15)
            current_month_mid = datetime(year, month, 15)
            total_days = (current_month_mid - prev_month_mid).days
            days_passed = (current_date - prev_month_mid).days
        else:
            current_month_mid = datetime(year, month, 15)
            next_month = month + 1
            next_year = year
            if next_month > 12:
                next_month = 1
                next_year += 1
            next_month_mid = datetime(next_year, next_month, 15)
            total_days = (next_month_mid - current_month_mid).days
            days_passed = (current_date - current_month_mid).days
        weight = max(0.0, min(1.0, days_passed / total_days)) if total_days > 0 else 0.0
        if day < 15:
            return (prev_year, prev_month), (year, month), weight
        else:
            return (year, month), (next_year, next_month), weight

    def load_hveg_data(self):
        if self.hveg_data is None and os.path.exists(self.hveg_path):
            self.hveg_data = self.read_mat_file(self.hveg_path, ['Hveg'])

    def process_xlsx_file(self, input_file_path):
        """处理 XLSX 文件，依次处理所有 sheet"""
        base, ext = os.path.splitext(input_file_path)
        output_file_path = f"{base}_ML.xlsx"

        try:
            xls = pd.ExcelFile(input_file_path)
            sheets = xls.sheet_names
            self.load_hveg_data()

            vod_vars = {
                'VOD_Ku_Hpol_Asc': 'ku_vod_H',
                'VOD_Ku_Vpol_Asc': 'ku_vod_V',
                'VOD_X_Hpol_Asc': 'x_vod_H',
                'VOD_X_Vpol_Asc': 'x_vod_V',
                'VOD_C_Hpol_Asc': 'c_vod_H',
                'VOD_C_Vpol_Asc': 'c_vod_V'
            }

            def generate_date_range(date_str, days_before=2, days_after=2):
                year = int(date_str[:4])
                month = int(date_str[4:6])
                day = int(date_str[6:8])
                base_date = datetime(year, month, day)
                return [(base_date + timedelta(days=i)).strftime('%Y%m%d') for i in range(-days_before, days_after+1)]

            with pd.ExcelWriter(output_file_path, engine='openpyxl') as writer:
                for sheet_name in sheets:
                    print(f"处理工作表: {sheet_name}")
                    df = pd.read_excel(input_file_path, sheet_name=sheet_name)

                    # 1. 计算行列索引
                    df['row'] = df.apply(lambda x: self.calculate_grid_indices(x['LonC'], x['LatC'])[0], axis=1)
                    df['col'] = df.apply(lambda x: self.calculate_grid_indices(x['LonC'], x['LatC'])[1], axis=1)

                    # 2. 日期列处理
                    df['Date'] = df['Date'].apply(self.safe_date_to_str)
                    df['YYYY'] = df['Date'].str[:4]
                    df['MM'] = df['Date'].str[4:6]
                    df['DD'] = df['Date'].str[6:8]

                    # 3. 初始化变量
                    for var in TARGET_VARIABLES:
                        df[var] = np.nan

                    # 4. VOD填充（逐变量处理5天均值）
                    for idx, row in df.iterrows():
                        date_str = row['Date']
                        if len(date_str) != 8:
                            continue
                        r, c = int(row['row']), int(row['col'])
                        for target_var, src_var in vod_vars.items():
                            valid_values = []
                            for d in generate_date_range(date_str):
                                year = int(d[:4])
                                file_path = os.path.join(
                                    self.vod_path,
                                    f"MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_{d}_V0.nc4.mat" if year <= 2012
                                    else f"MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_{d}_V0.nc4.mat"
                                )
                                if os.path.exists(file_path):
                                    if file_path not in self.vod_cache:
                                        self.vod_cache[file_path] = self.read_mat_file(file_path, list(vod_vars.values()), silent=True)
                                    vod_data = self.vod_cache[file_path]
                                    if vod_data and src_var in vod_data:
                                        val = vod_data[src_var][r, c]
                                        if not np.isnan(val) and val >= 0:
                                            valid_values.append(val)
                            if valid_values:
                                df.at[idx, target_var] = np.mean(valid_values)

                    # 5. PFT填充
                    pft_vars = {
                        'Grass_man': 'grassman',
                        'Grass_nat': 'grassnat',
                        'Shrub_bd': 'shrubbd',
                        'Shrub_be': 'shrubbe',
                        'Shrub_nd': 'shrubnd',
                        'Shrub_ne': 'shrubne',
                        'Tree_bd': 'treebd',
                        'Tree_be': 'treebe',
                        'Tree_nd': 'treend',
                        'Tree_ne': 'treene'
                    }
                    for year in df['YYYY'].unique():
                        file_path = os.path.join(self.pft_path, f"{year}.mat")
                        if os.path.exists(file_path):
                            if file_path not in self.pft_cache:
                                self.pft_cache[file_path] = self.read_mat_file(file_path, list(pft_vars.values()), silent=True)
                            pft_data = self.pft_cache[file_path]
                            if pft_data:
                                year_mask = df['YYYY'] == year
                                for target_var, src_var in pft_vars.items():
                                    if src_var in pft_data:
                                        for idx2 in df[year_mask].index:
                                            r = int(df.at[idx2, 'row'])
                                            c = int(df.at[idx2, 'col'])
                                            val = pft_data[src_var][r, c]
                                            if not np.isnan(val):
                                                df.at[idx2, target_var] = val

                    # 6. LAI填充
                    for idx, row in df.iterrows():
                        date_str = row['Date']
                        if len(date_str) != 8:
                            continue
                        prev_month, next_month, weight = self.calculate_lai_weight(date_str)
                        if prev_month is None:
                            continue
                        file1_path = os.path.join(self.lai_path, f"{prev_month[0]:04d}-{prev_month[1]:02d}-01.tif.mat")
                        file2_path = os.path.join(self.lai_path, f"{next_month[0]:04d}-{next_month[1]:02d}-01.tif.mat")
                        for fp in [file1_path, file2_path]:
                            if fp not in self.lai_cache:
                                self.lai_cache[fp] = self.read_mat_file(fp, ['lai'], silent=True) if os.path.exists(fp) else None
                        r, c = int(row['row']), int(row['col'])
                        lai1 = self.lai_cache[file1_path]['lai'][r, c] if self.lai_cache.get(file1_path) and 'lai' in self.lai_cache[file1_path] else np.nan
                        lai2 = self.lai_cache[file2_path]['lai'][r, c] if self.lai_cache.get(file2_path) and 'lai' in self.lai_cache[file2_path] else np.nan
                        if not np.isnan(lai1) and not np.isnan(lai2):
                            df.at[idx, 'LAI'] = (1 - weight) * lai1 + weight * lai2
                        elif not np.isnan(lai1):
                            df.at[idx, 'LAI'] = lai1
                        elif not np.isnan(lai2):
                            df.at[idx, 'LAI'] = lai2

                    # 7. Hveg填充
                    if self.hveg_data and 'Hveg' in self.hveg_data:
                        for idx, row in df.iterrows():
                            r, c = int(row['row']), int(row['col'])
                            val = self.hveg_data['Hveg'][r, c]
                            if not np.isnan(val):
                                df.at[idx, 'Hveg'] = val

                    # 写入 sheet
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"工作表 {sheet_name} 处理完成，共 {len(df)} 行")

            print(f"文件处理完成，已保存至: {output_file_path}")
            return True

        except Exception as e:
            print(f"处理文件时出错: {str(e)}")
            return False


# 简化接口
def process_xlsx_file(input_file_path, vod_path=None, pft_path=None, lai_path=None, hveg_path=None):
    processor = XLSXProcessor(vod_path, pft_path, lai_path, hveg_path)
    return processor.process_xlsx_file(input_file_path)