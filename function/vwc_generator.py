# vwc_generator.py
# VWC（植被含水率）单日数据生成工具（支持Hveg特征）
# ----------------------------------------------------------------------

import pandas as pd
import numpy as np
import time
import random
import os
import joblib
import sys
import h5py
import logging
import concurrent.futures
import warnings
from datetime import datetime, timedelta
from osgeo import gdal, osr
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# 忽略警告
warnings.filterwarnings('ignore')

# 设置随机种子保证可重复性
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


class VWCGenerator:
    """
    植被含水率（VWC）日影像生成类。
    封装了数据加载、特征准备、模型预测和GeoTIFF输出的整个流程。
    """
    def __init__(self, start_date, end_date, output_path, model_path, hveg_path,
                 vod_base_path, lai_base_path, pft_base_path, overwrite=False):
        
        # --- 配置变量 ---
        self.start_date = start_date
        self.end_date = end_date
        self.overwrite = overwrite
        self.output_path = output_path
        
        self.vod_base_path = vod_base_path
        self.lai_base_path = lai_base_path
        self.pft_base_path = pft_base_path
        self.hveg_path = hveg_path
        self.model_path = model_path
        
        # --- 状态变量 ---
        self.MODEL = None
        self.Hveg_DATA = None
        self.land_mask = None
        self.LAI_MONTH_CACHE = {} 
        self.PFT_YEAR_CACHE = {} 
        self.rows, self.cols = 1800, 3600 # 全球0.1度分辨率
        
        # --- 初始化 ---
        os.makedirs(self.output_path, exist_ok=True)
        self._configure_logging()
        self.logger.info("初始化 VWCGenerator...")
        self._load_model()
        self._load_hveg_data()
        self.land_mask = self._create_land_mask(self.rows, self.cols)
        self.logger.info(f"陆地掩膜创建完成: 有效点={np.count_nonzero(self.land_mask)}")

    def _configure_logging(self):
        """配置日志"""
        self.logger = logging.getLogger('VWCGenerator')
        self.logger.setLevel(logging.INFO)
        if not self.logger.handlers:
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            
            # 控制台输出
            ch = logging.StreamHandler()
            ch.setFormatter(formatter)
            self.logger.addHandler(ch)
            
            # 文件输出
            fh = logging.FileHandler(os.path.join(self.output_path, 'vwc_generation.log'))
            fh.setFormatter(formatter)
            self.logger.addHandler(fh)

    def _load_model(self):
        """加载模型"""
        try:
            if os.path.exists(self.model_path):
                self.MODEL = joblib.load(self.model_path)
                self.logger.info(f"成功加载模型: {self.model_path}")
                if hasattr(self.MODEL, 'feature_names_in_'):
                    self.logger.info(f"模型期望的特征顺序: {', '.join(self.MODEL.feature_names_in_)}")
            else:
                self.logger.error(f"模型文件不存在: {self.model_path}")
                sys.exit(1)
        except Exception as e:
            self.logger.error(f"加载模型失败: {str(e)}")
            sys.exit(1)

    def _load_hveg_data(self):
        """加载Hveg数据"""
        try:
            self.logger.info(f"加载Hveg数据: {self.hveg_path}")
            with h5py.File(self.hveg_path, 'r') as f:
                # 尝试不同可能的变量名
                for key in ['Hveg', 'CanopyHeight', 'CH']:
                    if key in f:
                        self.Hveg_DATA = np.array(f[key][:])
                        self.logger.info(f"成功加载Hveg数据: {key}")
                        break
                else:
                    keys = list(f.keys())
                    if keys:
                        self.Hveg_DATA = np.array(f[keys[0]][:])
                        self.logger.warning(f"使用默认数据集 '{keys[0]}' 作为Hveg数据")
                
                # 确保数据形状正确 (1800, 3600)
                if self.Hveg_DATA is not None:
                    if self.Hveg_DATA.shape == (self.cols, self.rows):
                        self.Hveg_DATA = self.Hveg_DATA.T
                    elif self.Hveg_DATA.shape != (self.rows, self.cols):
                        self.logger.error(f"不支持的Hveg数据形状: {self.Hveg_DATA.shape}")
                        self.Hveg_DATA = None
                
                if self.Hveg_DATA is not None:
                    self.logger.info(f"Hveg数据加载完成: {self.Hveg_DATA.shape}")
        except Exception as e:
            self.logger.error(f"加载Hveg数据失败: {str(e)}")
            self.Hveg_DATA = None

    # --- 辅助函数（大部分保留原逻辑，但使用 self.XXX 访问成员变量） ---

    def _create_singleband_geotiff(self, data, output_path, nodata=-9999.0):
        """创建单波段地理参考的TIFF文件"""
        try:
            driver = gdal.GetDriverByName('GTiff')
            rows, cols = data.shape
            
            out_ds = driver.Create(
                output_path, cols, rows, 1, gdal.GDT_Float32,
                options=['COMPRESS=LZW', 'BIGTIFF=YES']
            )
            
            # 设置地理变换 (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1)
            geotransform = (-180.0, 0.1, 0.0, 90.0, 0.0, -0.1)
            out_ds.SetGeoTransform(geotransform)
            
            # 设置坐标系 (WGS84)
            srs = osr.SpatialReference()
            srs.ImportFromEPSG(4326)
            out_ds.SetProjection(srs.ExportToWkt())
            
            band = out_ds.GetRasterBand(1)
            band.WriteArray(data)
            band.SetNoDataValue(nodata)
            band.SetDescription('VWC')
            
            out_ds.FlushCache()
            out_ds = None
            
            self.logger.info(f"成功创建GeoTIFF: {output_path}")
            return True
        except Exception as e:
            self.logger.error(f"创建GeoTIFF失败: {str(e)}")
            return False

    def _get_vod_file(self, date):
        """获取VOD文件路径"""
        date_str = date.strftime('%Y%m%d')
        possible_files = [
            f'MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_{date_str}_V0.nc4.mat',
            f'MCCA_AMSR2_010D_CCXH_VSM_VOD_Asc_{date_str}_V0.mat',
            f'MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_{date_str}_V0.nc4.mat',
            f'MCCA_AMSRE_010D_CCXH_VSM_VOD_Asc_{date_str}.mat'
        ]
        for filename in possible_files:
            file_path = os.path.join(self.vod_base_path, filename)
            if os.path.exists(file_path):
                return file_path
        self.logger.warning(f"未找到VOD文件: {date_str}")
        return None

    def _get_lai_file(self, year, month):
        """获取LAI文件路径"""
        month_str = str(month).zfill(2)
        possible_files = [
            f'{year}-{month_str}-01.tif.mat', f'{year}-{month_str}-01.mat',
            f'LAI_{year}{month_str}.mat', f'{year}_{month_str}_LAI.mat'
        ]
        for filename in possible_files:
            file_path = os.path.join(self.lai_base_path, filename)
            if os.path.exists(file_path):
                return file_path
        self.logger.warning(f"未找到LAI文件: {year}-{month_str}")
        return None

    def _get_pft_file(self, year):
        """获取PFT文件路径"""
        possible_files = [
            f'{year}.mat', f'PFT_{year}.mat', f'ESACCI_PFT_{year}.mat', f'pft_{year}.mat'
        ]
        for filename in possible_files:
            file_path = os.path.join(self.pft_base_path, filename)
            if os.path.exists(file_path):
                return file_path
        self.logger.warning(f"未找到PFT文件: {year}")
        return None

    def _get_month_centers(self, date):
        """获取用于插值的前后月中日期"""
        current_mid = date.replace(day=15)
        if date.day <= 15:
            prev_mid = (current_mid - timedelta(days=30)).replace(day=15)
            next_mid = current_mid
        else:
            prev_mid = current_mid
            next_mid = (current_mid + timedelta(days=30)).replace(day=15)
        return prev_mid, next_mid

    def _load_lai_matrix(self, file_path):
        """加载LAI矩阵数据"""
        if not file_path or not os.path.exists(file_path): return None
        try:
            with h5py.File(file_path, 'r') as f:
                for key in ['lai', 'Layer', 'data']:
                    if key in f:
                        lai_data = np.array(f[key][:])
                        break
                else:
                    keys = list(f.keys())
                    if keys:
                        lai_data = np.array(f[keys[0]][:])
                    else:
                        self.logger.warning(f"未找到数据集: {file_path}")
                        return None
                
                # 转置为(1800,3600)
                if lai_data.shape == (self.cols, self.rows):
                    lai_data = lai_data.T
                elif lai_data.shape != (self.rows, self.cols):
                    self.logger.error(f"不支持的LAI数据形状: {lai_data.shape}")
                    return None
                return lai_data
        except Exception as e:
            self.logger.error(f"加载LAI文件失败: {file_path} - {str(e)}")
            return None

    def _load_pft_matrix(self, file_path):
        """加载并旋转PFT矩阵数据"""
        if not file_path or not os.path.exists(file_path): return None
        try:
            pft_data = {}
            pft_mapping = { # 简化的PFT映射
                'grassnat': ['grassnat', 'grass_natural', 'GRASSNAT'], 'grassman': ['grassman', 'grass_managed', 'GRASSMAN'],
                'shrubbd': ['shrubbd', 'shrub_bd', 'SHRUBBD'], 'shrubbe': ['shrubbe', 'shrub_be', 'SHRUBBE'],
                'shrubnd': ['shrubnd', 'shrub_nd', 'SHRUBND'], 'shrubne': ['shrubne', 'shrub_ne', 'SHRUBNE'],
                'treebd': ['treebd', 'tree_bd', 'TREEBD'], 'treebe': ['treebe', 'tree_be', 'TREEBE'],
                'treend': ['treend', 'tree_nd', 'TREEND'], 'treene': ['treene', 'tree_ne', 'TREENE']
            }
            
            with h5py.File(file_path, 'r') as f:
                for target, aliases in pft_mapping.items():
                    for alias in aliases:
                        if alias in f:
                            data = np.array(f[alias][:])
                            if data.shape == (self.cols, self.rows * 2): # 0.05度到0.1度的降采样
                                data = np.rot90(data, k=-1)
                                data = (data[:, ::2] + data[:, 1::2]) / 2.0
                                pft_data[target] = data.T
                            elif data.shape == (self.cols, self.rows):
                                pft_data[target] = np.rot90(data, k=-1)
                            elif data.shape == (self.rows, self.cols):
                                pft_data[target] = data
                            else:
                                self.logger.warning(f"未知PFT形状: {data.shape} for {target}")
                                pft_data[target] = np.zeros((self.rows, self.cols))
                            break
                    else:
                        pft_data[target] = np.zeros((self.rows, self.cols))
            return pft_data
        except Exception as e:
            self.logger.error(f"加载PFT文件失败: {file_path} - {str(e)}")
            return None

    def _create_land_mask(self, rows, cols):
        """创建简单的陆地掩膜"""
        lats = np.linspace(90, -90, rows)
        land_mask = np.zeros((rows, cols), dtype=bool)
        for i in range(rows):
            if -60 <= lats[i] <= 80: # 排除南极和北极
                land_mask[i, :] = True
        return land_mask

    def _prepare_features(self, vod_data, lai_data, pft_data, hveg_data, valid_mask):
        """准备特征矩阵（6VOD + LAI + Hveg + PFTs）"""
        valid_indices = np.where(valid_mask)
        num_valid = len(valid_indices[0])
        if num_valid == 0:
            return None, None, None
        
        features = np.zeros((num_valid, 0), dtype=np.float32)
        feature_names = []
        
        # 1. VOD特征 (6个波段)
        vod_mat_mapping = { 'VOD_Ku_Hpol_Asc': 'ku_vod_H', 'VOD_Ku_Vpol_Asc': 'ku_vod_V',
                            'VOD_X_Hpol_Asc': 'x_vod_H', 'VOD_X_Vpol_Asc': 'x_vod_V',
                            'VOD_C_Hpol_Asc': 'c_vod_H', 'VOD_C_Vpol_Asc': 'c_vod_V' }
        
        for key, mat_key in vod_mat_mapping.items():
            feature_names.append(key)
            if key in vod_data:
                vod_val = vod_data[key] 
                features = np.column_stack((features, vod_val[valid_indices]))
            elif mat_key in vod_data: # 兼容MAT文件中的原始变量名
                vod_val = vod_data[mat_key] 
                features = np.column_stack((features, vod_val[valid_indices]))
            else:
                features = np.column_stack((features, np.zeros(num_valid)))

        # 2. LAI特征
        feature_names.append('LAI')
        if lai_data is not None:
            features = np.column_stack((features, lai_data[valid_indices]))
        else:
            features = np.column_stack((features, np.zeros(num_valid)))
        
        # 3. Hveg特征
        feature_names.append('Hveg')
        if hveg_data is not None:
            features = np.column_stack((features, hveg_data[valid_indices]))
        else:
            features = np.column_stack((features, np.zeros(num_valid)))
        
        # 4. PFT特征 (归一化)
        pft_features = [ 'Grass_man', 'Grass_nat', 'Shrub_bd', 'Shrub_be', 'Shrub_nd', 'Shrub_ne',
                         'Tree_bd', 'Tree_be', 'Tree_nd', 'Tree_ne' ]
        pft_key_mapping = { 'Grass_man': 'grassman', 'Grass_nat': 'grassnat', 'Shrub_bd': 'shrubbd', 
                            'Shrub_be': 'shrubbe', 'Shrub_nd': 'shrubnd', 'Shrub_ne': 'shrubne', 
                            'Tree_bd': 'treebd', 'Tree_be': 'treebe', 'Tree_nd': 'treend', 'Tree_ne': 'treene' }
        
        for feature in pft_features:
            feature_names.append(feature)
            pft_key = pft_key_mapping.get(feature)
            if pft_data and pft_key in pft_data:
                pft_val = pft_data[pft_key] / 100.0 # PFT特征归一化
                features = np.column_stack((features, pft_val[valid_indices]))
            else:
                features = np.column_stack((features, np.zeros(num_valid)))
        
        self.logger.debug(f"准备特征完成: {features.shape} 形状, 特征数: {len(feature_names)}")
        return features, feature_names, valid_indices

    def _predict_vwc(self, features, feature_names, rows, cols, valid_mask):
        """预测VWC"""
        if features is None:
            return None
        
        vwc_array = np.full((rows, cols), -9999.0, dtype=np.float32)
        valid_indices = np.where(valid_mask)
        num_valid = len(valid_indices[0])
        
        if num_valid == 0: return vwc_array
        
        try:
            # 检查和重新排序特征以匹配模型期望
            if hasattr(self.MODEL, 'feature_names_in_'):
                model_features = list(self.MODEL.feature_names_in_)
                if feature_names != model_features:
                    try:
                        missing_features = set(model_features) - set(feature_names)
                        if missing_features:
                            self.logger.error(f"缺失特征: {', '.join(missing_features)}")
                            return None
                        
                        sorted_indices = [feature_names.index(f) for f in model_features]
                        features = features[:, sorted_indices]
                        feature_names = [feature_names[i] for i in sorted_indices]
                        self.logger.debug("特征重新排序完成")
                    except Exception as e:
                        self.logger.error(f"特征重新排序失败: {str(e)}")
                        return None
            
            # 预测 (分批处理)
            predictions = np.zeros(num_valid, dtype=np.float32)
            chunk_size = 500000 
            chunks = (num_valid + chunk_size - 1) // chunk_size
            
            for chunk_idx in tqdm(range(chunks), desc=f"模型预测({self.MODEL.__class__.__name__})", leave=False):
                start = chunk_idx * chunk_size
                end = min((chunk_idx + 1) * chunk_size, num_valid)
                X_chunk = features[start:end]
                predictions[start:end] = self.MODEL.predict(X_chunk)
            
            vwc_array[valid_indices] = predictions
            self.logger.info(f"预测值统计: min={np.min(predictions):.4f}, max={np.max(predictions):.4f}, mean={np.mean(predictions):.4f}")
            return vwc_array
            
        except Exception as e:
            self.logger.error(f"预测失败: {str(e)}")
            return None

    def _process_one_date(self, date):
        """处理单日数据"""
        output_filename = f'VWC-{date.strftime("%Y%m%d")}.tif'
        output_path = os.path.join(self.output_path, output_filename)
        
        if os.path.exists(output_path) and not self.overwrite:
            self.logger.info(f"文件已存在: {output_path} - 跳过")
            return True
        
        if os.path.exists(output_path):
            try: os.remove(output_path)
            except Exception as e:
                self.logger.error(f"删除文件失败: {output_path} - {str(e)}")
                return False
        
        self.logger.info(f"处理日期: {date.strftime('%Y-%m-%d')}")
        
        try:
            # 1. 加载VOD数据
            vod_file = self._get_vod_file(date)
            if not vod_file or not os.path.exists(vod_file):
                self.logger.error(f"VOD文件未找到: {date}")
                return False
                
            vod_data = {}
            vod_mat_mapping = { 'ku_vod_H': 'VOD_Ku_Hpol_Asc', 'ku_vod_V': 'VOD_Ku_Vpol_Asc',
                                'x_vod_H': 'VOD_X_Hpol_Asc', 'x_vod_V': 'VOD_X_Vpol_Asc',
                                'c_vod_H': 'VOD_C_Hpol_Asc', 'c_vod_V': 'VOD_C_Vpol_Asc' }
            
            with h5py.File(vod_file, 'r') as f:
                for key in f.keys():
                    if key in vod_mat_mapping or key.lower() in vod_mat_mapping:
                        data = np.array(f[key][:])
                        if data.shape == (self.cols, self.rows):
                             data = data.T # 确保形状一致
                        vod_data[key] = data
                        
                qc_data = np.array(f['QC'][:,:]).T
            
            # 2. 创建有效掩膜
            qc_mask = qc_data == 0
            valid_mask = qc_mask & self.land_mask
            
            # 3. 加载LAI（插值和缓存）
            prev_mid, next_mid = self._get_month_centers(date)
            lai_prev_data = self._get_cached_lai(prev_mid)
            lai_next_data = self._get_cached_lai(next_mid)
            
            if lai_prev_data is None or lai_next_data is None: return False
            
            total_days = (next_mid - prev_mid).days
            current_offset = (date - prev_mid).days
            weight = current_offset / total_days
            lai_data = lai_prev_data * (1 - weight) + lai_next_data * weight
            lai_data = np.nan_to_num(lai_data, nan=0.0)
            lai_mask = ~np.isnan(lai_data) 
            valid_mask = valid_mask & lai_mask
            
            # 4. 加载PFT（缓存）
            pft_data = self._get_cached_pft(date.year)
            if pft_data is None: return False
            
            # 5. 准备特征
            num_valid = np.count_nonzero(valid_mask)
            self.logger.info(f"有效数据点数量: {num_valid}")
            
            features, feature_names, _ = self._prepare_features(
                vod_data, lai_data, pft_data, self.Hveg_DATA, valid_mask
            )
            
            if features is None or num_valid == 0:
                self.logger.warning("无有效数据点，跳过预测")
                return True
            
            # 6. 预测VWC
            vwc_array = self._predict_vwc(features, feature_names, self.rows, self.cols, valid_mask)
            
            if vwc_array is None:
                self.logger.error("VWC预测失败")
                return False
            
            # 7. 创建GeoTIFF
            return self._create_singleband_geotiff(vwc_array, output_path)
            
        except Exception as e:
            self.logger.error(f"处理日期 {date} 错误: {str(e)}", exc_info=True)
            return False

    def _get_cached_lai(self, date):
        """获取缓存的LAI数据"""
        year, month = date.year, date.month
        key = (year, month)
        if key in self.LAI_MONTH_CACHE:
            return self.LAI_MONTH_CACHE[key]
        
        lai_file = self._get_lai_file(year, month)
        lai_data = self._load_lai_matrix(lai_file)
        if lai_data is None:
            self.logger.error(f"LAI文件加载失败: {year}-{month:02d}")
            return None
        self.LAI_MONTH_CACHE[key] = lai_data
        return lai_data

    def _get_cached_pft(self, year):
        """获取缓存的PFT数据"""
        if year in self.PFT_YEAR_CACHE:
            return self.PFT_YEAR_CACHE[year]
        
        pft_file = self._get_pft_file(year)
        pft_data = self._load_pft_matrix(pft_file)
        if pft_data is None:
            self.logger.error(f"PFT文件加载失败: {year}")
            return None
        self.PFT_YEAR_CACHE[year] = pft_data
        return pft_data
    
    # ------------------------------------------------
    # --- 公共方法：生成VWC影像的主入口 ---
    # ------------------------------------------------
    def generate_vwc_maps(self, max_workers=4):
        """生成VWC影像（并行优化版本）"""
        
        dates = [self.start_date + timedelta(days=i) 
                 for i in range((self.end_date - self.start_date).days + 1)]
        
        futures = []
        completed, failed = 0, 0
        
        self.logger.info(f"开始并行处理{len(dates)}天数据，最大线程数: {max_workers}")
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for date in dates:
                # 提交任务，注意这里调用的是实例方法，会自动访问 self.land_mask, self.MODEL 等
                futures.append(executor.submit(self._process_one_date, date))
            
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="生成VWC影像"):
                try:
                    result = future.result()
                    if result:
                        completed += 1
                    else:
                        failed += 1
                except Exception as e:
                    self.logger.error(f"任务失败: {str(e)}", exc_info=True)
                    failed += 1
        
        # 清理缓存
        self.LAI_MONTH_CACHE.clear()
        self.PFT_YEAR_CACHE.clear()
        
        self.logger.info(f"处理完成: {completed}天成功, {failed}天失败")
        return completed, failed


# ============================== 可直接运行的入口 ==============================

def run_vwc_generation(start_date_str, end_date_str, overwrite=False, max_workers=4, **path_kwargs):
    """
    主运行函数，可以直接在其他Python文件或命令行中调用。

    :param start_date_str: 开始日期，格式 'YYYY-MM-DD'
    :param end_date_str: 结束日期，格式 'YYYY-MM-DD'
    :param overwrite: 是否覆盖已存在文件 (bool)
    :param max_workers: 并行处理的线程数 (int)
    :param path_kwargs: 包含所有路径配置的字典。
        - output_path: 输出目录 (str)
        - model_path: 模型文件路径 (str)
        - hveg_path: Hveg数据路径 (str)
        - vod_base_path: VOD数据基础路径 (str)
        - lai_base_path: LAI数据基础路径 (str)
        - pft_base_path: PFT数据基础路径 (str)
    """
    try:
        start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
        end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    except ValueError:
        print("错误: 日期格式必须是 'YYYY-MM-DD'")
        return 0, 0

    # 检查所有必要的路径参数
    required_paths = ['output_path', 'model_path', 'hveg_path', 'vod_base_path', 'lai_base_path', 'pft_base_path']
    for key in required_paths:
        if key not in path_kwargs:
            print(f"错误: 缺少必要的路径参数 '{key}'")
            return 0, 0

    try:
        generator = VWCGenerator(
            start_date=start_date,
            end_date=end_date,
            overwrite=overwrite,
            **path_kwargs # 传入所有路径参数
        )
        
        print(f"开始处理: {start_date.strftime('%Y-%m-%d')} 到 {end_date.strftime('%Y-%m-%d')}")
        completed, failed = generator.generate_vwc_maps(max_workers=max_workers)
        print(f"VWC影像生成完成: 成功 {completed} 天, 失败 {failed} 天")
        return completed, failed
        
    except Exception as e:
        print(f"程序运行错误: {e}")
        return 0, 0
