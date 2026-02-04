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
matplotlib.rcParams['font.family'] = 'Times New Roman'        # 全局英文字体设为 Times New Roman
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
        "Vegetation water content(kg/m²)",  # 去掉 "(Provided by Field Experiment)"
        fontsize=20, 
        fontweight='bold',
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

def plot_landcover_with_overlays_consistent(
    landcover_tif_path,
    output_path,
    overlay_tif_list=None,          # 可选：需要画红色矩形框的影像tif列表
    station_marker="^",             # 默认保持与你原函数一致
    station_file=None,              # 可选：Site.xlsx
    sheet_name=None,                # 用于标题；读取站点时若该sheet不存在自动回退到第一个sheet
    tick_step=1,                    # ✅ 新增：经纬度绘制间距（默认1°）
    overlay_valid_min=0,            # 保留参数但本版本不用于边界提取（仅用于未来兼容）
    draw_colorbar=False             # 建议：主图默认不画IGBP色标
):
    """
    使用 landcover 分类 tif (值 1-17) 作为底图，按固定配色绘制；
    对 overlay_tif_list 中每张影像：直接读取其经纬度边界(extent)，绘制红色矩形框（不提取mask边界）；
    站点使用 Site.xlsx（经度/纬度列）叠加点位；
    tick_step 控制经纬度标注间距（度）。

    ✅ 改进点（本版本）：
    1) 影像边界：只画矩形框（extent），不做不规则mask轮廓提取。
    2) 显示范围：每张图的显示范围强制与背景 landcover 的范围完全相同（set_extent=landcover_extent）。
    3) 读取站点Excel时：若指定 sheet_name 不存在，自动读取第一个 sheet。
    4) 站点无效值过滤：删除经纬度为 -99 的记录，并按范围过滤：
       - 纬度在 [-90, 90]
       - 经度在 [-180, 180]
    5) landcover 值域过滤：仅保留 1-17，否则设为 NaN。
    """

    if overlay_tif_list is None:
        overlay_tif_list = []

    # ---------- IGBP配色（1-17） ----------
    igbp_colors = [
        '#106919', '#3fb334', '#54e30d', '#2ca25f', '#93cc31',
        '#e3818e', '#ebca79', '#ffaa01', '#ffd380', '#cdf57a',
        '#00ffcc', '#f5f579', '#c90d01', '#9e9933', '#ffffff',
        '#cccbc7', '#41b1df'
    ]
    cmap = mcolors.ListedColormap(igbp_colors)
    bounds = np.arange(0.5, 17.5 + 1e-9, 1.0)
    norm = mcolors.BoundaryNorm(bounds, cmap.N)

    # ---------- 工具：读取栅格 extent ----------
    def _read_raster_extent(path):
        with rasterio.open(path) as src:
            t = src.transform
            extent = [
                t[2],                                  # left
                t[2] + t[0] * src.width,               # right
                t[5] + t[4] * src.height,              # bottom
                t[5]                                   # top
            ]
        return extent

    # ---------- 工具：安全读取站点（优先sheet_name，否则第一张表） ----------
    def _read_site_safe(station_file, sheet_name=None):
        ext = os.path.splitext(station_file)[-1].lower()
        if ext == ".csv":
            df = pd.read_csv(station_file)
        elif ext in [".xls", ".xlsx"]:
            if sheet_name is not None:
                try:
                    df = pd.read_excel(station_file, sheet_name=sheet_name)
                except ValueError:
                    df = pd.read_excel(station_file, sheet_name=0)
            else:
                df = pd.read_excel(station_file, sheet_name=0)

            if isinstance(df, dict):
                df = df[list(df.keys())[0]]
        else:
            raise ValueError("station_file 必须是 CSV 或 Excel 文件")
        return df

    # ---------- 工具：站点清洗（去-99 & 越界） ----------
    def _clean_sites(df):
        if ('经度' not in df.columns) or ('纬度' not in df.columns):
            return None
        sdf = df.dropna(subset=['经度', '纬度']).copy()

        # 去除 -99
        sdf = sdf[(sdf['经度'] != -99) & (sdf['纬度'] != -99)].copy()

        # 范围过滤：经度[-180,180], 纬度[-90,90]
        sdf = sdf[sdf['经度'].between(-180, 180) & sdf['纬度'].between(-90, 90)].copy()

        return sdf

    # ---------- 读 landcover ----------
    with rasterio.open(landcover_tif_path) as src:
        lc = src.read(1)
        t = src.transform
        landcover_extent = [
            t[2],
            t[2] + t[0] * src.width,
            t[5] + t[4] * src.height,
            t[5]
        ]
        nodata = src.nodata
        if nodata is not None:
            lc = np.where(lc == nodata, np.nan, lc)

    # ✅ landcover 值域过滤（仅保留1-17）
    lc = np.where(np.isfinite(lc) & (lc >= 1) & (lc <= 17), lc, np.nan)

    # ---------- 读站点（可选） ----------
    stations_df = None
    if station_file is not None:
        try:
            raw = _read_site_safe(station_file, sheet_name=sheet_name)
            stations_df = _clean_sites(raw)
            if stations_df is None or len(stations_df) == 0:
                print("⚠️ 站点表无有效点（已自动去除-99与越界），将不绘制站点。")
                stations_df = None
        except Exception as e:
            print(f"⚠️ 站点读取/清洗失败（不影响绘图）：{e}")
            stations_df = None

    # ---------- 绘图（版式与原函数一致） ----------
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())

    # 底图 landcover
    img = ax.imshow(
        lc,
        cmap=cmap,
        norm=norm,
        extent=landcover_extent,
        transform=ccrs.PlateCarree(),
        origin="upper",
        interpolation="nearest",
        zorder=1
    )

    # overlays：画红色矩形框（extent）
    for p in overlay_tif_list:
        try:
            e = _read_raster_extent(p)
            left, right, bottom, top = e[0], e[1], e[2], e[3]

            # 画矩形框：按经纬度四个角闭合
            xs = [left, right, right, left, left]
            ys = [bottom, bottom, top, top, bottom]
            ax.plot(
                xs, ys,
                color='black',
                linewidth=2.0,
                transform=ccrs.PlateCarree(),
                zorder=5
            )
        except Exception as ex:
            print(f"⚠️ 读取overlay范围失败：{p} -> {ex}")

    # stations：样式完全对齐原函数（红色点+黑边）
    if stations_df is not None:
        ax.scatter(
            stations_df['经度'],
            stations_df['纬度'],
            s=120,
            c='red',
            marker=station_marker,
            edgecolor='black',
            linewidth=1,
            transform=ccrs.PlateCarree(),
            zorder=10
        )

    # ✅ 显示范围：强制与背景 landcover 完全一致
    ax.set_extent(landcover_extent, crs=ccrs.PlateCarree())

    # 经纬度格网（字号/样式对齐原函数，只是locator步长可控）
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
    gl.xlocator = mticker.MultipleLocator(tick_step)
    gl.ylocator = mticker.MultipleLocator(tick_step)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {'size': 16, 'color': 'black'}
    gl.ylabel_style = {'size': 16, 'color': 'black'}

    # 标题（对齐原函数）
    if sheet_name is not None:
        ax.set_title(sheet_name, fontsize=24, pad=20, fontweight='bold')

    # IGBP色标：默认不画（按你的建议）；若画则尽量对齐原函数风格
    if draw_colorbar:
        cbar = fig.colorbar(
            img,
            ax=ax,
            orientation='horizontal',
            fraction=0.045,
            pad=0.12
        )
        cbar.ax.set_title(
            "IGBP land cover class",
            fontsize=20,
            fontweight='bold',
            pad=6
        )
        ticks = np.arange(1, 18, 1)
        cbar.set_ticks(ticks)
        cbar.ax.set_xticklabels([str(t) for t in ticks])
        cbar.ax.tick_params(labelsize=16, length=0)
        cbar.outline.set_visible(False)

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 图像已保存: {output_path}（landcover底图 + 矩形框叠加，tick_step={tick_step}）")

def plot_igbp_legend_from_tifs(
    landcover_tif_list,
    output_path,
    ncols=6,                 # ✅ 强制6列（默认6）
    title="IGBP Land Cover Types",
    col_gap=0.00,            # ✅ 列间距（axes比例），建议 0.01~0.05
    text_pad=0.06            # ✅ 色块与文字间距（按col_w比例），建议 0.03~0.12
):
    """
    输入为IGBP影像集合（tif列表，像元值1-17），提取所有出现的类别值，
    绘制一个独立legend图：6列布局，每项左侧颜色块矩形，右侧地物类型简写。

    参数
    ----------
    landcover_tif_list : list[str]
        IGBP landcover tif 路径列表（像元值1-17）
    output_path : str
        输出 PNG 路径
    ncols : int
        legend 强制列数（默认6）
    title : str or None
        标题；传 None 或 "" 则不绘制标题
    col_gap : float
        列间距（axes坐标系比例），越大列间距越大
    text_pad : float
        色块到文字间距（按col_w比例）
    """

    igbp_colors = [
        '#106919', '#3fb334', '#54e30d', '#2ca25f', '#93cc31',
        '#e3818e', '#ebca79', '#ffaa01', '#ffd380', '#cdf57a',
        '#00ffcc', '#f5f579', '#c90d01', '#9e9933', '#ffffff',
        '#cccbc7', '#41b1df'
    ]

    igbp_abbr = [
        "ENF", "EBF", "DNF", "DBF", "MF",
        "CSH", "OSH", "WSA", "SAV", "WET",
        "GRA", "CRO", "CVM", "BAR", "WAT",
        "SNO", "URB"
    ]

    # ---- 统计出现的类别值（1-17）----
    classes = set()
    for p in landcover_tif_list:
        with rasterio.open(p) as src:
            arr = src.read(1)
            nd = src.nodata
            if nd is not None:
                arr = np.where(arr == nd, np.nan, arr)

        vals = np.unique(arr[np.isfinite(arr)])
        # 有些tif可能是float，转int前先过滤
        for v in vals:
            try:
                iv = int(v)
                if 1 <= iv <= 17:
                    classes.add(iv)
            except Exception:
                continue

    classes = sorted(classes)

    if len(classes) == 0:
        raise ValueError("未在输入的 landcover tif 中检测到 1-17 的有效类别值。")

    # ---- 6列布局 ----
    n = len(classes)
    ncols = int(ncols)
    nrows = int(np.ceil(n / ncols))

    # 画布大小：根据行数自适应（保持清晰）
    fig_w = max(8, ncols * 2.2)
    fig_h = max(2.5, nrows * 0.7 + 1.2)

    fig = plt.figure(figsize=(fig_w, fig_h))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    # title：与主图一致（24 bold）
    # 传 None 或 "" 不绘制
    if title:
        ax.text(
            0.03, 0.95, title,
            ha="left", va="top",
            fontsize=24, fontweight="bold",
            transform=ax.transAxes
        )

    # 布局参数（axes坐标系里排版）
    top_y = 0.85
    left_x = 0.03
    right_x = 0.97
    bottom_y = 0.08

    # ✅ 行高
    row_h = (top_y - bottom_y) / nrows

    # ✅ 列宽：扣掉列间距 col_gap*(ncols-1)
    # 注意：col_gap 过大可能导致 col_w <= 0
    col_w = (right_x - left_x - col_gap * (ncols - 1)) / ncols
    if col_w <= 0:
        raise ValueError(f"col_gap 过大导致列宽为非正数：col_w={col_w}。请减小 col_gap。")

    # 色块大小（相对单元格）
    rect_w = col_w * 0.20
    rect_h = row_h * 0.55

    for i, cls in enumerate(classes):
        r = i // ncols
        c = i % ncols  # ✅ c 在循环内定义

        # 单元格左上角坐标（加入列间距）
        x0 = left_x + c * (col_w + col_gap)
        y0 = top_y - (r + 1) * row_h

        # 色块
        rect = plt.Rectangle(
            (x0, y0 + (row_h - rect_h) / 2),
            rect_w, rect_h,
            facecolor=igbp_colors[cls - 1],
            edgecolor="black",
            linewidth=0.6,
            transform=ax.transAxes
        )
        ax.add_patch(rect)

        # 文本：缩写（字号16，与主图刻度字号一致）
        ax.text(
            x0 + rect_w + col_w * text_pad,
            y0 + row_h / 2,
            igbp_abbr[cls - 1],
            ha="left", va="center",
            fontsize=20,
            transform=ax.transAxes
        )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ IGBP legend 已保存: {output_path}（包含类别: {classes}）")


def plot_overlay_station_legend(
    output_path,
    box_color='black',
    station_color='red',
    station_edgecolor='black',
    station_marker='^',
    box_linewidth=1.6,
    station_size=240,
    title=None
):
    """
    单独输出“影像方框 + 实测站点”的legend图像（两列排版）。
    - 不依赖任何tif/站点文件
    - 风格与现有绘图保持一致（Times New Roman 已在全局 rcParams 设置）
    - 两列：左列为影像方框样例，右列为站点样例

    参数
    ----------
    output_path : str
        输出PNG路径
    box_color : str
        影像方框颜色（默认黑色）
    station_color : str
        站点填充颜色（默认红色）
    station_edgecolor : str
        站点描边颜色（默认黑色）
    station_marker : str
        站点marker（默认 '^'，与主函数默认一致）
    box_linewidth : float
        方框线宽（默认 1.6，与主函数一致）
    station_size : int
        站点大小（默认 30，与主函数一致）
    title : str or None
        可选标题
    """

    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D

    # 画布：只用于legend，不需要地图投影
    fig = plt.figure(figsize=(8, 2.5))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis('off')

    if title is not None:
        ax.set_title(title, fontsize=24, pad=10, fontweight='bold')

    # ---- 定义两个legend项 ----
    # 1) 影像方框：用空心矩形示意
    box_handle = Rectangle(
        (0, 0),
        1, 1,
        fill=False,
        edgecolor=box_color,
        linewidth=box_linewidth
    )

    # 2) 站点：用scatter风格的Line2D做legend句柄
    station_handle = Line2D(
        [0], [0],
        marker=station_marker,
        linestyle='None',
        markersize=np.sqrt(station_size),  # legend里用markersize近似对应scatter的s
        markerfacecolor=station_color,
        markeredgecolor=station_edgecolor,
        markeredgewidth=1
    )

    # 两列排版
    handles = [box_handle, station_handle]
    labels = ["Remote-sensing product extent", "In situ sites"]

    leg = ax.legend(
        handles, labels,
        ncol=2,
        loc='center',
        frameon=False,
        fontsize=24,          # 与主图经纬度标签字号一致
        handlelength=2.0,
        columnspacing=1.0,
        handletextpad=0.4,
        borderaxespad=0.0
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ legend 图像已保存: {output_path}")


