import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="武大地空一号电离层观测数据处理系统", page_icon="🛰️", layout="wide")
st.markdown(
    """
    <style>
    /* 1. 设置星空背景图 */
    .stApp {
        background-image: url("https://images.unsplash.com/photo-1451187580459-43490279c0fa?q=80&w=2072&auto=format&fit=crop");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }
    
    /* 2. 加一层黑色半透明遮罩，防止背景太亮影响看字 */
    .stApp::before {
        content: "";
        position: absolute;
        top: 0; left: 0; width: 100%; height: 100%;
        background-color: rgba(0, 0, 0, 0.7); 
        z-index: -1;
    }
    </style>
    """,
    unsafe_allow_html=True
)
# ========== 读取数据（兼容本地上传与云端部署） ==========
@st.cache_data
def load_data(uploaded_file):
    # 使用 h5netcdf 引擎，它支持直接读取 Streamlit 的文件流对象，不依赖本地硬盘
    ds = xr.open_dataset(uploaded_file, engine="h5netcdf")
    return ds

# ========== 画图函数 ==========
def plot_scatter(sp_lon_data, sp_lat_data, current_lon, current_lat, energy_data):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use('dark_background') 
    
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)
    
    # 设置图表背景为半透明深色（透出星空，但不影响看字）
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.8)   # 80% 不透明
    ax.patch.set_facecolor('black')
    ax.patch.set_alpha(0.8)
    
    # 严谨计算颜色映射上下限
    valid_energy = energy_data[energy_data > 0]
    v_min = np.percentile(valid_energy, 2)
    v_max = np.percentile(valid_energy, 98)
    if v_min >= v_max:
        v_max = v_min + 1e-20

    # 绘制散点图
    scatter = ax.scatter(sp_lon_data, sp_lat_data, s=5, c=energy_data, 
                         cmap='viridis', alpha=0.8, 
                         vmin=v_min, vmax=v_max, 
                         label="历史观测点")
    
    # 添加颜色条
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Signal Peak Power', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # 标出当前选中的点
    ax.scatter(current_lon, current_lat, s=250, color='red', marker='*', 
               edgecolor='white', linewidth=1, label='当前选中点', zorder=5)
    
    # 美化图表
    ax.set_title("CYGNSS 真实观测点地理分布与能量映射", fontsize=16, pad=15)
    ax.set_xlabel("经度 (Longitude)", fontsize=12)
    ax.set_ylabel("纬度 (Latitude)", fontsize=12)
    ax.grid(True, linestyle="--", alpha=0.4, color='gray')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    
    ax.legend(loc="upper right", fontsize=11, framealpha=0.8)
    
    return fig

def plot_ddm(DDM_data):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(6, 6), dpi=120)
    
    # 设置图表背景为半透明深色
    fig.patch.set_facecolor('black')
    fig.patch.set_alpha(0.8)
    
    img = plt.imshow(DDM_data, cmap='jet', origin='lower', aspect='auto', interpolation='nearest')
    
    ax = plt.gca()
    ax.patch.set_facecolor('black')
    ax.patch.set_alpha(0.8)
    
    cbar = plt.colorbar(img)
    cbar.set_label('Signal Power', fontsize=12)
    plt.title('真实 DDM 热力图', fontsize=14, pad=15)
    plt.xlabel('多普勒频移(Doppler Bins)', fontsize=14)
    plt.ylabel('延迟通道(Delay Bins)', fontsize=14)
    
    plt.tight_layout()
    return fig

# ========== 侧边栏配置与文件上传 ==========
st.sidebar.title("🛰️ 系统控制面板")
st.sidebar.markdown("---")

# 核心：文件上传器
uploaded_file = st.sidebar.file_uploader("请上传 CYGNSS Level 1 数据 (.nc)", type=["nc"])

st.sidebar.markdown("---")
app_mode = st.sidebar.radio(
    "请选择功能模块：",
    ["🏠 系统首页", "📊 双图联动大屏", "🔍 信号特征提取"]
)
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 武大地空一号大创项目组")

# ========== 主界面区 ==========
st.markdown("<h1 style='white-space: nowrap; color: #1E90FF;'>🛰️ 武大地空一号 - 电离层观测数据处理系统</h1>", unsafe_allow_html=True)

# 核心判断：如果用户还没上传文件，就显示提示并停止运行
if uploaded_file is None:
    st.warning("⚠️ 请先在左侧控制面板上传真实的 `.nc` 数据文件！")
    st.stop() # 停止运行，等待上传
else:
    # 如果上传了文件，就加载数据
    ds = load_data(uploaded_file)
    st.success("✅ 状态：数据加载成功！")
    
    # 提取基础经纬度数据
    lon = ds["sp_lon"].values[:, 0]
    lon = np.where(lon > 180, lon - 360, lon)
    lat = ds["sp_lat"].values[:, 0]
    total_samples = len(lon) - 1 # 动态获取样本总数

    # =========== 页面渲染逻辑 ==========
    if app_mode == "🏠 系统首页":
        st.markdown("### 欢迎使用！") 
        st.info("请在左侧控制面板选择需要执行的操作模式。")
        st.markdown("### 📁 当前数据集信息")
        st.write(f"**观测项目:** {ds.attrs.get('project', 'CYGNSS')}")
        # 增加安全读取，防止某些文件没有这些属性报错
        st.write(f"**观测起始时间:** {ds.attrs.get('time_coverage_start', '未知')}")
        st.write(f"**观测结束时间:** {ds.attrs.get('time_coverage_end', '未知')}")
        st.write(f"**总观测样本量:** {total_samples} 条")

    else:
        # 如果不是首页，侧边栏出现滑动条
        st.sidebar.markdown("### ⚙ 观测参数")
        # 动态根据数据长度设置滑块最大值
        sample_idx = st.sidebar.slider("🎛️ 调节观测样本序号", 0, total_samples, min(5000, total_samples))
        
        # 提取当前样本的DDM数据
        DDM_data = ds["power_analog"].values[sample_idx, :, :]
        if len(DDM_data.shape) > 2:
            DDM_data = DDM_data[0, :, :]
        all_energy=np.max(ds["power_analog"].values[:,0,:,:],axis=(1,2))

        if app_mode == "📊 双图联动大屏":
            st.subheader("📊 宏观轨迹与微观信号联动大屏")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌏 真实观测点全球轨迹")
                fig = plot_scatter(lon, lat, lon[sample_idx], lat[sample_idx],all_energy)
                st.pyplot(fig)

                #添加当前样本信息点
                st.markdown("---")
                current_lon=lon[sample_idx]
                current_lat=lat[sample_idx]
                current_time=str(ds["ddm_timestamp_utc"].values[sample_idx])
                st.markdown(f"📍 **当前选中坐标：** 经度 {current_lon:.2f}° | 纬度 {current_lat:.2f}°")
                st.markdown(f"🕒 **观测时间：** {current_time}")      
            with col2:
                st.subheader(f"📡 DDM热力图 (sample{sample_idx})")
                fig = plot_ddm(DDM_data)
                st.pyplot(fig)
                st.info(f"当前展示的是第 {sample_idx} 号样本的 DDM 矩阵")   

        elif app_mode == "🔍 信号特征提取":
            st.subheader(f"🔍 样本 {sample_idx} 电离层特征提取")
            
            # 1. 找最高点（仅供展示）
            max_idx = np.unravel_index(np.argmax(DDM_data, axis=None), DDM_data.shape)
            max_val = np.max(DDM_data)
            total_energy = np.sum(DDM_data)
            
            # 2. 核心：计算整张矩阵的“能量加权质心”
            rows, cols = DDM_data.shape
            row_indices = np.arange(rows)
            col_indices = np.arange(cols)
            
            # 计算每一行的总能量，以及每一列的总能量
            row_energy_sum = np.sum(DDM_data, axis=1)
            col_energy_sum = np.sum(DDM_data, axis=0)
            
            # 质心公式：(坐标 * 能量) 的总和 / 总能量
            center_delay = np.sum(row_indices * row_energy_sum) / total_energy
            center_doppler = np.sum(col_indices * col_energy_sum) / total_energy
            
            # 3. 计算最终偏移量（最强点 减去 质心）
            delay_offset = max_idx[0] - center_delay
            doppler_offset = max_idx[1] - center_doppler
            
            # 4. 计算前沿斜率 (LES)
            peak_delay, peak_dopp = max_idx
            if 0 < peak_dopp < cols - 1:
                les = abs(DDM_data[peak_delay, peak_dopp+1] - DDM_data[peak_delay, peak_dopp-1])
            else:
                les = 0.0
            
            # --- 显示指标卡片 ---
            col_a, col_b, col_c = st.columns(3)
            # 用 :+.2f 会强制显示正负号，看起来更专业
            col_a.metric(label="延迟偏移量 (bins)", value=f"{delay_offset:+.2f}")
            col_b.metric(label="多普勒偏移量 (bins)", value=f"{doppler_offset:+.2f}")
            col_c.metric(label="前沿斜率 (LES)", value=f"{les:.2e}")
            
            st.markdown("---")
            
            # --- 物理说明 ---
            st.markdown("### 💡 核心物理特征说明")
            st.caption("*基准算法：基于全矩阵能量分布的二维加权质心 (Center of Mass)*")
            
            st.write("- **延迟/多普勒偏移量**：我们计算了整片 DDM 散射面的能量质心。最强点偏离质心的距离，直观反映了电离层自由电子导致的信号拖尾畸变程度。正值代表向高延迟/高频移方向偏移。")
            st.write("- **前沿斜率 (LES)**：波形前沿的陡峭程度（已取绝对值），对电离层局部的电子密度梯度极其敏感。")
            
            # 画出 DDM 图
            st.pyplot(plot_ddm(DDM_data))
            st.success(f"系统定位：信号能量质心位于 (延迟 {center_delay:.2f}, 多普勒 {center_doppler:.2f})。")







     


    
