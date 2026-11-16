import streamlit as st
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

st.set_page_config(page_title="武大地空一号电离层观测数据处理系统", page_icon="🛰️", layout="wide")

# ========== 1. 读取数据（兼容本地上传与云端部署） ==========
@st.cache_data
def load_data(uploaded_file):
    # 使用 h5netcdf 引擎，它支持直接读取 Streamlit 的文件流对象，不依赖本地硬盘
    ds = xr.open_dataset(uploaded_file, engine="h5netcdf")
    return ds

# ========== 2. 画图函数 ==========
def plot_scatter(sp_lon_data, sp_lat_data, current_lon, current_lat):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 6))
    plt.style.use('dark_background')
    plt.scatter(sp_lon_data, sp_lat_data, s=10, color="dodgerblue", alpha=0.6, label="历史观测点")
    plt.scatter(current_lon, current_lat, s=200, color='red', marker='*', label='当前选中点')
    plt.title("CYGNSS 真实观测点地理分布轨迹", fontsize=16)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.xlabel("经度")
    plt.ylabel("纬度")
    plt.legend()
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    return plt.gcf()

def plot_ddm(DDM_data):
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(6, 6), dpi=120)
    plt.style.use('dark_background')
    img = plt.imshow(DDM_data, cmap='jet', origin='lower', aspect='auto', interpolation='nearest')
    cbar = plt.colorbar(img)
    cbar.set_label('Signal Power', fontsize=12)
    plt.title('真实 DDM 热力图', fontsize=14, pad=15)
    plt.xlabel('多普勒频移(Doppler Bins)', fontsize=14)
    plt.ylabel('延迟通道(Delay Bins)', fontsize=14)
    plt.tight_layout()
    return plt.gcf()

# ========== 3. 侧边栏配置与文件上传 ==========
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

# ========== 4. 主界面区 ==========
st.markdown("<h1 style='white-space: nowrap; color: #1E90FF;'>🛰️ 武大地空一号 - 电离层观测数据处理系统</h1>", unsafe_allow_html=True)

# 核心判断：如果用户还没上传文件，就显示提示并停止运行
if uploaded_file is None:
    st.warning("⚠️ 请先在左侧控制面板上传真实的 `.nc` 数据文件！")
    st.info("💡 提示：如果你本地没有数据，可以在 NASA Earthdata 官网下载 CYGNSS Level 1 数据，或者向队长索要样例文件。")
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

    # ================= 页面渲染逻辑 =================
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
        
        # 提取当前样本的 DDM
        DDM_data = ds["power_analog"].values[sample_idx, :, :]
        if len(DDM_data.shape) > 2:
            DDM_data = DDM_data[0, :, :]

        if app_mode == "📊 双图联动大屏":
            st.subheader("📊 宏观轨迹与微观信号联动大屏")
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("🌏 真实观测点全球轨迹")
                fig = plot_scatter(lon, lat, lon[sample_idx], lat[sample_idx])
                st.pyplot(fig)        
            with col2:
                st.subheader(f"📡 DDM热力图 (sample{sample_idx})")
                fig = plot_ddm(DDM_data)
                st.pyplot(fig)
                st.info(f"当前展示的是第 {sample_idx} 号样本的 DDM 矩阵")   

        elif app_mode == "🔍 信号特征提取":
            st.subheader(f"🔍 样本 {sample_idx} 物理特征自动提取")
            max_idx = np.unravel_index(np.argmax(DDM_data, axis=None), DDM_data.shape)
            total_energy = np.sum(DDM_data)
            
            col_a, col_b, col_c = st.columns(3)
            col_a.metric(label="镜面点延迟通道", value=f"{max_idx[0]}")
            col_b.metric(label="镜面点多普勒通道", value=f"{max_idx[1]}")
            col_c.metric(label="总反射能量", value=f"{total_energy:.2e}")
            
            st.markdown("---")
            st.pyplot(plot_ddm(DDM_data))
            st.info("注：图表中的最高亮区域，即系统自动定位的镜面反射点。")







     


    
