import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.animation import FuncAnimation
from datetime import datetime, timedelta, timezone
import matplotlib.colors as mcolors
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import matplotlib as mpl
import random
import math
import os
from PIL import Image

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 自定义台风强度颜色映射
def create_typhoon_colormap():
    colors = [
        (0.0, "#3498db"),    # 热带低压 - 蓝色
        (0.2, "#2ecc71"),    # 热带风暴 - 绿色
        (0.4, "#f1c40f"),    # 强热带风暴 - 黄色
        (0.6, "#e67e22"),    # 台风 - 橙色
        (0.8, "#e74c3c"),    # 强台风 - 红色
        (1.0, "#c0392b")     # 超强台风 - 深红色
    ]
    return LinearSegmentedColormap.from_list("typhoon_intensity", colors)

# 随机台风名称生成器
def generate_typhoon_name():
    prefixes = ["台风"]
    names = ["Yanjing", "Touxiang", "Maoenenjiao", "Huiuiwang", "Chuoxia", "Rua", 
            "Haixiuxiu","140", "1500","Oeo","Ooewa","Dachuan","Caibaoni","Nekochan", "Ciji", "Dili", "Chawan", "Ainvren", "Aihenhenai"]
    return random.choice(prefixes) + random.choice(names)

# 检查点是否在陆地上
def is_point_on_land(lat, lon):
    # 定义主要陆地区域（纬度，经度范围）
    land_areas = [
        (20, 40, 90, 125),    # 中国
        (30, 45, 125, 145),    # 日本
        (5, 20, 115, 130),     # 菲律宾
        (10, 20, 90, 110),    # 东南亚
        (22, 25, 120, 122)     # 台湾
    ]
    
    for area in land_areas:
        min_lat, max_lat, min_lon, max_lon = area
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return True
    return False

# 随机台风路径生成
def generate_random_typhoon_data(typhoon_id, existing_typhoons=[]):
    # 获取当前北京时间（UTC+8）
    now_beijing = datetime.now(timezone(timedelta(hours=8)))
    
    # 计算最近的3倍数小时（向下取整）
    current_hour = now_beijing.hour
    recent_3hour = (current_hour // 3) * 3
    
    # 创建最近的3倍数时间作为基准
    base_time = datetime(now_beijing.year, now_beijing.month, now_beijing.day, recent_3hour, 0, tzinfo=timezone(timedelta(hours=8)))
    base_time = base_time.replace(tzinfo=None)  # 移除时区信息
    
    # 随机起始时间偏移（0-7天）
    start_time_offset = random.randint(0, 7)
    start_time = base_time + timedelta(days=start_time_offset)


 
    # 随机起始位置（西太平洋区域），确保不在陆地上
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        start_lat = random.uniform(4.0, 28.0)
        start_lon = random.uniform(105.0, 180.0)
        
        # 检查是否在陆地上
        if not is_point_on_land(start_lat, start_lon):
            # 检查与其他台风的距离
            too_close = False
            for typhoon in existing_typhoons:
                dist = math.sqrt((start_lat - typhoon['latitude'][0])**2 + 
                                (start_lon - typhoon['longitude'][0])**2)
                if dist < 15:  # 如果距离小于15度，重新生成位置
                    too_close = True
                    break
            
            if not too_close:
                break
        
        attempt += 1
    
    if attempt >= max_attempts:
        # 如果多次尝试后仍无法找到合适位置，使用默认海洋位置
        start_lat = random.uniform(10.0, 20.0)
        start_lon = random.uniform(130.0, 160.0)

        
    # 随机路径长度（3-30天）
    num_points = random.randint(24, 240)
    
    # 随机台风名称
    name = generate_typhoon_name()

    # 初始化路径数据
    lats = [start_lat]
    lons = [start_lon]
    times = [start_time]
    
    # 随机初始强度
    wind_speeds = [random.uniform(15, 25)]
    
    # 随机移动方向控制点
    num_control_points = random.randint(3, 6)
    control_points = []
    current_index = 0
    
    for i in range(num_control_points):
        # 随机阶段长度
        phase_length = random.randint(5, num_points//num_control_points)
        current_index += phase_length
        
        # 随机移动方向（纬度变化率，经度变化率）
        lat_move = random.uniform(-0.05, 0.1)
        lon_move = random.uniform(-0.08, 0.1)  # 更可能向西移动
        
        # 避免超出合理范围
        current_index = min(current_index, num_points-1)
        control_points.append((current_index, lat_move, lon_move))
    
    # 随机巅峰次数（1-7次）
    num_peaks = random.randint(1, 7)
    peak_times = []
    
    if num_peaks > 0:
        # 在路径中随机选择巅峰位置
        peak_indices = sorted(random.sample(range(10, num_points-10), num_peaks))
        peak_times = peak_indices
    
   # 生成路径 
    current_phase = 0
    for i in range(1, num_points):
        # 检查是否进入下一个阶段
        if current_phase < len(control_points) and i > control_points[current_phase][0]:
            current_phase += 1
        
        # 获取当前阶段的移动参数
        if current_phase < len(control_points):
            lat_move = control_points[current_phase][1]
            lon_move = control_points[current_phase][2]
        else:
            # 如果没有更多控制点，继续使用最后一个
            lat_move = control_points[-1][1] if control_points else 0
            lon_move = control_points[-1][2] if control_points else 0

        # 添加随机扰动 - 修改以减少南移倾向
        # 纬度扰动：更倾向于向北
        lat_perturb = random.uniform(-0.02, 0.08)  # 减少南移扰动

       # 经度扰动：更倾向于向西
        lon_perturb = random.uniform(-0.02, 0.05)  # 减少向东扰动
        
        lat = lats[-1] + lat_move + lat_perturb
        lon = lons[-1] + lon_move + lon_perturb
        time = times[-1] + timedelta(hours=3)  # 每3小时一个点
        
        # 确保位置在合理范围内
        lat = max(5.0, min(lat, 40.0))  # 限制在5-40度纬度之间
        lon = max(100, min(lon, 180))
        
        # 模拟台风强度变化 - 修改以确保在陆地上减弱
        ws = wind_speeds[-1]
        
        # 基础变化趋势 - 增加增强的可能性
        base_change = random.uniform(0.0, 1.5)  # 增强的概率更大
        
        # 计算当前点在路径中的进度（0-1）
        progress = i / num_points

        if random.random() < 0.7:  # 70%概率加强西北向
            lat_move = min(0.4, lat_move + 0.05)  # 略微增加北移
            lon_move = max(-0.6, lon_move - 0.05)  # 略微增加西移
        
        current_index = min(current_index, num_points-1)
        control_points.append((current_index, lat_move, lon_move))


        # 添加额外的方向控制 - 路径中后期更倾向西北
        if i > num_points * 0.4:  # 40%路径后
            lat_perturb = max(-0.01, lat_perturb)  # 几乎不南移
            lon_perturb = min(-0.02, lon_perturb)  # 几乎不西移
        
        lat = lats[-1] + lat_move + lat_perturb
        lon = lons[-1] + lon_move + lon_perturb

        
        # 如果在巅峰点附近，增强变化幅度
        is_near_peak = any(abs(i - p) < 5 for p in peak_times)
        if is_near_peak:
            # 在巅峰点附近增强
            if i < max(peak_times):
                # 显著增强
                base_change = random.uniform(1.0, 4.0)
            else:
                # 巅峰后减弱
                base_change = random.uniform(-2.5, -1.0)
        
        # 判断是否在陆地上 - 使用更精确的陆地判断
        is_on_land = is_point_on_land(lat, lon)

        # 如果接近陆地或在陆地上，强度减弱
        if is_on_land:
            # 在陆地上强制减弱
            base_change = random.uniform(-2.5, 0)
            # 如果之前是巅峰期，减弱更快
            if is_near_peak:
                base_change = random.uniform(-3.5, -1.5)
        else:
            # 如果接近陆地（海岸线附近），可能减弱
            distance_to_land = min(abs(lat-20), abs(lat-25), abs(lat-30))  # 简化计算
            if distance_to_land < 5 and random.random() < 0.7:
                base_change = random.uniform(-1.0, 0.5)  # 减弱幅度减小
        
        # 如果接近其他台风，可能产生藤原效应
        for typhoon in existing_typhoons:
            if typhoon['id'] != typhoon_id:  # 排除自己
                # 计算与其他台风的距离
                idx = min(i, len(typhoon['latitude']) - 1)
                other_lat = typhoon['latitude'][idx]
                other_lon = typhoon['longitude'][idx]
                
                dist = math.sqrt((lat - other_lat)**2 + (lon - other_lon)**2)
                if dist < 20:  # 如果距离小于20度，可能产生互旋
                    # 藤原效应：相互旋转
                    angle = math.atan2(other_lat - lat, other_lon - lon)
                    # 添加旋转分量
                    lat += 0.1 * math.sin(angle)
                    lon += 0.1 * math.cos(angle)
                    # 强度减弱幅度
                    base_change = random.uniform(-0.5, 0.5)
        
        # 在路径最后阶段（最后20%）强制更大幅度减弱
        if progress > 0.8:
            base_change = random.uniform(-2.5, -1.0)  # 增大减弱幅度
            if is_on_land:
                base_change = random.uniform(-3.5, -1.5)  # 陆地上减弱更快
        
        ws += base_change
        ws = max(10, min(ws, 80))
        
        lats.append(lat)
        lons.append(lon)
        times.append(time)
        wind_speeds.append(ws)
    
    # 确保台风在最后减弱到热带低压或以下（15m/s以下）
    if wind_speeds[-1] > 15.0:
        # 最后10个点强制减弱
        for j in range(1, min(11, len(wind_speeds))):
            index = len(wind_speeds) - j
            wind_speeds[index] = max(10, wind_speeds[index] - random.uniform(1.5, 3.5))
            if is_point_on_land(lats[index], lons[index]):
                wind_speeds[index] = max(10, wind_speeds[index] - random.uniform(2.0, 4.0))
        
        # 确保最后一个点确实低于15m/s
        wind_speeds[-1] = min(15.0, wind_speeds[-1])
    
    return {
        'id': typhoon_id,
        'name': name,
        'latitude': np.array(lats),
        'longitude': np.array(lons),
        'time': np.array(times),
        'wind_speed': np.array(wind_speeds),
        'peak_times': np.array(peak_times),
        'color': plt.cm.tab10(typhoon_id)
    }

# 获取强度等级描述
def get_intensity_category(wind_speed):
    if wind_speed < 17.5:
        return "热带低压"
    elif wind_speed < 24.5:
        return "热带风暴"
    elif wind_speed < 32.7:
        return "强热带风暴"
    elif wind_speed < 41.5:
        return "台风"
    elif wind_speed < 51.0:
        return "强台风"
    else:
        return "超强台风"

# 创建东亚地图
def create_map():
    fig = plt.figure(figsize=(14, 10), dpi=100)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    
    # 添加地理特征
    ax.add_feature(cfeature.LAND, facecolor='#f0f0f0')
    ax.add_feature(cfeature.OCEAN, facecolor='#d0e0ff')
    ax.add_feature(cfeature.COASTLINE, linewidth=0.8)
    ax.add_feature(cfeature.BORDERS, linestyle=':', linewidth=0.8)
    ax.add_feature(cfeature.STATES, linestyle=':', linewidth=0.5)
    ax.add_feature(cfeature.LAKES, facecolor='#d0e0ff', edgecolor='none')
    ax.add_feature(cfeature.RIVERS, edgecolor='#a0c0ff', linewidth=0.8)
    
    # 设置地图范围（东亚和西太平洋）
    ax.set_extent([90, 180, 0, 50])
    
    # 添加网格线
    gl = ax.gridlines(draw_labels=True, linestyle='--', alpha=0.7)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    
    return fig, ax

# 生成路径方向描述
def get_path_description(lons, lats):
    if len(lons) < 2:
        return "稳定"
    
    # 计算总体方向
    delta_lon = lons[-1] - lons[0]
    delta_lat = lats[-1] - lats[0]
    
    # 计算角度（度数）
    angle = math.degrees(math.atan2(delta_lat, delta_lon))
    
    if angle < -67.5:
        return "西南向"
    elif angle < -22.5:
        return "西向"
    elif angle < 22.5:
        return "西北向"
    elif angle < 67.5:
        return "北向"
    elif angle < 112.5:
        return "东北向"
    else:
        return "东南向"

    # 主函数
def main():
    # 创建保存GIF的目录
    if not os.path.exists("huihuiwang_typhoon_gifs"):
        os.makedirs("huihuiwang_typhoon_gifs")
    
    # 随机生成1-3个台风
    num_typhoons = random.randint(1, 3)
    typhoons = []
    
    # 生成台风数据
    for i in range(num_typhoons):
        typhoons.append(generate_random_typhoon_data(i, typhoons))
    
    # 创建地图
    fig, ax = create_map()
    
    # 获取当前北京时间
    beijing_time = datetime.now(timezone.utc) + timedelta(hours=8)
    time_str = beijing_time.strftime('%Y-%m-%d %H:%M')
    
    # 添加北京时间文本（右上角）
    time_text = ax.text(0.98, 0.95, f'北京时间: {time_str}', 
                       transform=ax.transAxes, fontsize=11, ha='right',
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                                 boxstyle='round,pad=0.5'))
    
    # 创建台风强度颜色映射
    cmap = create_typhoon_colormap()
    norm = plt.Normalize(vmin=10, vmax=90)  # 调整范围以匹配新的风速范围
    
    # 初始化每个台风的图形元素
    path_lines = []
    current_points = []
    intensity_points = []
    peak_markers_list = []
    info_texts = []
    
    # 为每个台风创建图形元素
    for i, typhoon in enumerate(typhoons):
        # 路径线
        path_line, = ax.plot([], [], '-', lw=1.5, alpha=0.6, 
                            color=typhoon['color'], label=typhoon['name'])
        path_lines.append(path_line)
        
        # 当前点
        current_point = ax.scatter([], [], s=150, edgecolors='black', 
                                  linewidths=1.2, zorder=10, 
                                  color=typhoon['color'])
        current_points.append(current_point)
        
        # 强度点
        intensity_point = ax.scatter([], [], s=40, alpha=0.7)
        intensity_points.append(intensity_point)
        
        # 巅峰标记
        peak_markers = []
        for j, peak in enumerate(typhoon['peak_times']):
            marker = ax.scatter([], [], s=100, marker='*', color='gold', 
                               edgecolors='black', zorder=20, visible=False)
            peak_markers.append(marker)
        peak_markers_list.append(peak_markers)
        
        # 信息文本（每个台风一个）
        # 调整位置，确保有足够空间显示时间行
        info_text = ax.text(0.02, 0.85 - i*0.12, '', transform=ax.transAxes, 
                           fontsize=9,  # 减小字体大小以适应更多行
                           bbox=dict(facecolor='white', alpha=0.8, 
                           edgecolor=typhoon['color'], boxstyle='round,pad=0.3'))
        info_texts.append(info_text)


    # 随机选择情绪
    emotions = ["(羡慕)", "(不羡慕)"]
    emotion = random.choice(emotions)

        # 标题
    title_text = ax.text(0.5, 0.95, f' {num_typhoons}个台风{emotion}', 
                        transform=ax.transAxes, fontsize=14, ha='center',
                        bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray', 
                                  boxstyle='round,pad=0.5'))
    
    # 添加强度图例
    cax = fig.add_axes([0.15, 0.08, 0.25, 0.02])  # [left, bottom, width, height]
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax, 
                        orientation='horizontal')
    cbar.set_label('风速 (m/s)', fontsize=9)
    cbar.ax.tick_params(labelsize=8)
    
    # 添加版权信息
    credit_text = ax.text(0.98, 0.02, '数据来源: Huihui Wang',
                         transform=ax.transAxes, fontsize=8, 
                         ha='right', alpha=0.7)
    
    # 计算最大帧数（所有台风的最大路径长度）
    max_frames = max(len(typhoon['time']) for typhoon in typhoons)
    
    # 动画更新函数
    def update(frame):
        # 更新每个台风
        all_artists = [title_text, credit_text, time_text]
        
        for i, typhoon in enumerate(typhoons):
            # 当前帧在台风路径中的索引
            current_idx = min(frame, len(typhoon['time']) - 1)
            
            # 更新路径
            path_lines[i].set_data(typhoon['longitude'][:current_idx+1], 
                                  typhoon['latitude'][:current_idx+1])
            
            # 更新当前点
            if current_idx >= 0:
                # 当前点位置和颜色
                current_points[i].set_offsets(
                    np.c_[typhoon['longitude'][current_idx], 
                    typhoon['latitude'][current_idx]]
                )
                current_points[i].set_facecolor(cmap(norm(typhoon['wind_speed'][current_idx])))
                
                # 设置当前点大小（根据风速）
                size = 80 + typhoon['wind_speed'][current_idx] * 1.5
                current_points[i].set_sizes([size])
                
                # 更新所有点的颜色
                intensity_points[i].set_offsets(
                    np.c_[typhoon['longitude'][:current_idx+1], 
                          typhoon['latitude'][:current_idx+1]]
                )
                intensity_points[i].set_array(typhoon['wind_speed'][:current_idx+1])
                intensity_points[i].set_cmap(cmap)
                intensity_points[i].set_norm(norm)
                
                # 获取强度等级
                category = get_intensity_category(typhoon['wind_speed'][current_idx])
                
                # 获取当前点的时间
                current_time = typhoon['time'][current_idx]
                # 格式化时间字符串
                time_str = current_time.strftime('%m-%d %H:%M')
                
                # 更新台风信息（添加时间行）
                info_text = (f"{typhoon['name']}\n"
                             f"时间: {time_str}\n"
                             f"强度: {category}\n"
                             f"风速: {typhoon['wind_speed'][current_idx]:.1f} m/s\n"
                             f"位置: {typhoon['latitude'][current_idx]:.1f}°N, "
                             f"{typhoon['longitude'][current_idx]:.1f}°E")
                
                info_texts[i].set_text(info_text)
                
                # 更新巅峰标记
                for j, peak in enumerate(typhoon['peak_times']):
                    if current_idx == peak:
                        peak_markers_list[i][j].set_offsets(
                            np.c_[typhoon['longitude'][current_idx], 
                                  typhoon['latitude'][current_idx]]
                        )
                        peak_markers_list[i][j].set_visible(True)
                        # 添加巅峰标记文本
                        peak_text = ax.text(typhoon['longitude'][current_idx], 
                                           typhoon['latitude'][current_idx] + 1.5,
                                           f"{typhoon['name'][-2:]}-巅峰{j+1}",
                                           fontsize=8, ha='center', 
                                           bbox=dict(facecolor='white', alpha=0.8, 
                                           edgecolor=typhoon['color']))
                        all_artists.append(peak_text)
        
        # 收集所有需要更新的艺术家对象
        all_artists.extend(path_lines)
        all_artists.extend(current_points)
        all_artists.extend(intensity_points)
        all_artists.extend(info_texts)
        
        # 添加巅峰标记
        for markers in peak_markers_list:
            all_artists.extend(markers)
        
        return all_artists
    
       # 创建动画
    ani = FuncAnimation(fig, update, frames=max_frames,
                        interval=150, blit=True)
    
    # 调整布局
    plt.subplots_adjust(top=0.90, bottom=0.15, left=0.05, right=0.95)
    
    # 生成GIF文件名
    typhoon_names = '-'.join([t['name'][-2:] for t in typhoons])
    gif_filename = f"huihuiwang_typhoon_gifs/multi_typhoon_{typhoon_names}.gif"
    
    # 保存动画
    print(f"正在保存GIF动画: {gif_filename}")
    ani.save(gif_filename, writer='pillow', fps=8, dpi=100)
    print("GIF保存完成!")
    
    # 显示完成信息
    plt.close()
    print(f"成功生成{num_typhoons}个台风的动态路径:")
    for i, typhoon in enumerate(typhoons):
        peak_times = [typhoon['time'][p].strftime('%m-%d') for p in typhoon['peak_times']]
        max_wind = max(typhoon['wind_speed'])
        max_category = get_intensity_category(max_wind)
        final_wind = typhoon['wind_speed'][-1]
        final_category = get_intensity_category(final_wind)
        start_time = typhoon['time'][0].strftime('%m-%d %H:%M')
        end_time = typhoon['time'][-1].strftime('%m-%d %H:%M')
        print(f"{i+1}. {typhoon['name']}: {len(typhoon['time'])}个点位")
        print(f"   起始时间: {start_time}, 结束时间: {end_time}")
        print(f"   最大强度: {max_category}({max_wind:.1f}m/s), 巅峰时间: {', '.join(peak_times)}")
        print(f"   最终强度: {final_category}({final_wind:.1f}m/s)")
    
    # 显示生成的GIF
    img = Image.open(gif_filename)
    img.show()

if __name__ == '__main__':
    main()
