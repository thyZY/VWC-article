import rasterio
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pandas as pd
import numpy as np
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import os
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
matplotlib.rcParams['font.family'] = 'Arial'        # 全局英文字体设为 Arial
matplotlib.rcParams['axes.unicode_minus'] = False   # 避免负号显示异常

def plot_vwc_with_stations(
    tif_path, 
    output_path, 
    station_marker="^",  # 自定义站点形状
    station_file=None, 
    sheet_name=None, 
    vmin=0,              # ✅ 新增：颜色条下限
    vmax=20              # ✅ 新增：颜色条上限
):
    """
    绘制地表覆盖TIF影像 (imshow)，并可选绘制站点数据 (scatter)，
    带1°经纬度格网与连续颜色条，可自定义颜色条范围。
    
    参数
    ----------
    tif_path : str
        输入的VWC TIF文件路径。
    output_path : str
        输出的PNG图像路径。
    station_file : str, optional
        Excel或CSV文件路径，需包含 '经度' 和 '纬度' 列。
    sheet_name : str, optional
        Excel的sheet名称。
    station_marker : str, optional
        matplotlib的marker字符串，默认 '^'。
    vmin, vmax : float, optional
        自定义颜色条最小值和最大值（默认0-20）。
    """
    # -------- 读取 TIF --------
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform
        extent = [
            transform[2],
            transform[2] + transform[0] * src.width,
            transform[5] + transform[4] * src.height,
            transform[5]
        ]
        
        if src.nodata is not None:
            data = np.where(data == src.nodata, np.nan, data)
        
        # 屏蔽无效值：小于等于 0 的地方不显示颜色
        data = np.where(data <= 0, np.nan, data)
    
    # -------- 自定义 colormap --------
    colors = [
        '#fe3c19',  # 最低值
        '#ffac18',
        '#f2fe2a',
        '#7cb815',
        '#147218'   # 最高值
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list("custom_vwc", colors)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # -------- 绘图 --------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # ax.add_feature(cfeature.COASTLINE, linewidth=0.5)
    # ax.add_feature(cfeature.BORDERS, linewidth=0.5, linestyle="--")
    
    img = ax.imshow(
        data, 
        cmap=cmap, 
        norm=norm,
        extent=extent,
        transform=ccrs.PlateCarree(),
        origin="upper",
        interpolation="nearest"
    )
    
    # -------- 可选：绘制站点 --------
    if station_file is not None:
        try:
            ext = os.path.splitext(station_file)[-1].lower()
            if ext == ".csv":
                df = pd.read_csv(station_file)
            elif ext in [".xls", ".xlsx"]:
                df = pd.read_excel(station_file, sheet_name=sheet_name)
                if isinstance(df, dict):
                    first_key = list(df.keys())[0]
                    df = df[first_key]
            else:
                raise ValueError("station_file 必须是 CSV 或 Excel 文件")
            
            if '经度' in df.columns and '纬度' in df.columns:
                ax.scatter(
                    df['经度'], 
                    df['纬度'], 
                    s=30, 
                    c='red', 
                    marker=station_marker,
                    edgecolor='black', 
                    linewidth=1,
                    transform=ccrs.PlateCarree(),
                    zorder=5
                )
            else:
                print("⚠️ 文件中缺少 '经度' 和 '纬度' 列，未绘制站点。")
        except Exception as e:
            print(f"⚠️ 读取站点数据失败: {e}")
    
    # -------- 地图范围 --------
    ax.set_extent([extent[0], extent[1], extent[2], extent[3]], crs=ccrs.PlateCarree())
    
    # -------- 经纬度格网 --------
    gl = ax.gridlines(
        crs=ccrs.PlateCarree(), 
        draw_labels=True, 
        linewidth=0.5, 
        color='gray', 
        alpha=0.7, 
        linestyle='--'
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlocator = mticker.MultipleLocator(1)
    gl.ylocator = mticker.MultipleLocator(1)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    gl.ylabel_style = {'size': 16, 'color': 'black'}

    # -------- 添加标题（居中，字体大小24）--------
    if sheet_name is not None:
        ax.set_title(sheet_name, fontsize=24, pad=20, fontweight='bold') 
           
    # -------- 颜色条 --------
    cbar = fig.colorbar(
        img, 
        ax=ax, 
        orientation='horizontal', 
        fraction=0.045,  # 控制颜色条长度
        pad=0.12         # 控制颜色条与图像间距
    )
    cbar.ax.set_title(
        "VWC Synchronous Retrieval Product (kg/m²)",  # 去掉 "(Provided by Field Experiment)"
        fontsize=20, 
        pad=6  # 控制标题与颜色条的距离，可调节为 4-10 之间
    )
    
    # ✅ 自动生成 5 个刻度点（等间距）
    ticks = np.linspace(vmin, vmax, 5)
    cbar.set_ticks(ticks)
    cbar.ax.set_xticklabels([f"{t:.1f}" for t in ticks])
    cbar.ax.tick_params(labelsize=16, length=0)
    cbar.outline.set_visible(False)
    
    # -------- 保存 --------
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 图像已保存: {output_path}（颜色条范围: {vmin}–{vmax}）")


import os
import pandas as pd

def extract_unique_coords(file_path, output_path, sheet_name=None):
    """
    从 xlsx/csv 文件中提取唯一的经纬度对，并保存为新的表格文件。
    支持覆盖已有文件时保留其他工作表。
    """
    # -------- 读取文件 --------
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".xlsx":
        try:
            if sheet_name is not None:
                # 尝试读取指定 sheet
                df = pd.read_excel(file_path, sheet_name=sheet_name)
            else:
                df = pd.read_excel(file_path)
        except ValueError:
            # 如果 sheet_name 不存在，则读取第一个 sheet
            df = pd.read_excel(file_path, sheet_name=0)
    elif ext == ".csv":
        df = pd.read_csv(file_path)
    else:
        raise ValueError("仅支持 xlsx 或 csv 文件")
    
    # -------- 检查列名 --------
    if not {"Latitude", "Longitude"}.issubset(df.columns):
        raise ValueError("输入表格必须包含 'Latitude' 和 'Longitude' 两列")
    
    # -------- 提取唯一经纬度 --------
    coords = df[["Longitude", "Latitude"]].drop_duplicates().reset_index(drop=True)
    coords = coords.rename(columns={"Longitude": "经度", "Latitude": "纬度"})
    
    # -------- 创建输出目录 --------
    out_dir = os.path.dirname(output_path)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    # -------- 输出 Excel 或 CSV --------
    out_ext = os.path.splitext(output_path)[-1].lower()
    
    if out_ext == ".xlsx":
        # 确定输出工作表名称
        output_sheet = sheet_name if sheet_name is not None else "Sheet1"
        
        # 处理文件已存在的情况
        if os.path.exists(output_path):
            # 读取现有文件
            with pd.ExcelWriter(output_path, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                # 删除同名工作表（如果存在）
                if output_sheet in writer.book.sheetnames:
                    std = writer.book[output_sheet]
                    writer.book.remove(std)
                
                # 写入新数据
                coords.to_excel(writer, sheet_name=output_sheet, index=False)
        else:
            # 文件不存在，直接写入
            coords.to_excel(output_path, sheet_name=output_sheet, index=False)
        
        print(f"✅ 已保存唯一经纬度表格，共 {len(coords)} 条记录 -> {output_path} ({output_sheet})")
    
    elif out_ext == ".csv":
        coords.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"✅ 已保存唯一经纬度表格，共 {len(coords)} 条记录 -> {output_path}")
    
    else:
        raise ValueError("输出文件必须是 xlsx 或 csv")


