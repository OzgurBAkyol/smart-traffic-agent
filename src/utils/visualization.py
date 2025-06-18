import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def plot_traffic_simulation(env, container):
    """Trafik simülasyonunu görselleştirir"""
    # Kesişim noktası koordinatları
    intersection = {
        'x': [0, 0, 1, 1],
        'y': [0, 1, 1, 0]
    }
    
    # Yolların koordinatları
    roads = {
        'E2C': {'x': [-1, 0], 'y': [0.5, 0.5]},
        'W2C': {'x': [1, 0], 'y': [0.5, 0.5]},
        'N2C': {'x': [0.5, 0.5], 'y': [1, 0]},
        'S2C': {'x': [0.5, 0.5], 'y': [-1, 0]}
    }
    
    # Araçların konumlarını hesapla
    vehicles = []
    for edge_id in ['E2C', 'W2C', 'N2C', 'S2C']:
        count = env.vehicle_counts[edge_id]
        queue = env.queue_lengths[edge_id]
        
        # Yol üzerindeki araçları yerleştir
        for i in range(count):
            if edge_id == 'E2C':
                x = -0.1 - (i * 0.1)
                y = 0.5
            elif edge_id == 'W2C':
                x = 1.1 + (i * 0.1)
                y = 0.5
            elif edge_id == 'N2C':
                x = 0.5
                y = 1.1 + (i * 0.1)
            else:  # S2C
                x = 0.5
                y = -0.1 - (i * 0.1)
            
            vehicles.append({
                'x': x,
                'y': y,
                'color': 'red' if env.traffic_light_states[edge_id] == 'red' else 'green'
            })
    
    # Grafik oluştur
    fig = go.Figure()
    
    # Kesişim noktasını çiz
    fig.add_trace(go.Scatter(
        x=intersection['x'],
        y=intersection['y'],
        fill='toself',
        fillcolor='lightgray',
        line=dict(color='black'),
        name='Kesişim'
    ))
    
    # Yolları çiz
    for road_id, road in roads.items():
        fig.add_trace(go.Scatter(
            x=road['x'],
            y=road['y'],
            line=dict(color='gray', width=5),
            name=f'Yol {road_id}'
        ))
    
    # Araçları çiz
    for vehicle in vehicles:
        fig.add_trace(go.Scatter(
            x=[vehicle['x']],
            y=[vehicle['y']],
            mode='markers',
            marker=dict(
                size=10,
                color=vehicle['color'],
                symbol='circle'
            ),
            name='Araç'
        ))
    
    # Trafik ışıklarını göster
    for edge_id, state in env.traffic_light_states.items():
        if edge_id in ['E2C', 'W2C', 'N2C', 'S2C']:
            if edge_id == 'E2C':
                x, y = -0.2, 0.5
            elif edge_id == 'W2C':
                x, y = 1.2, 0.5
            elif edge_id == 'N2C':
                x, y = 0.5, 1.2
            else:  # S2C
                x, y = 0.5, -0.2
            
            fig.add_trace(go.Scatter(
                x=[x],
                y=[y],
                mode='markers',
                marker=dict(
                    size=15,
                    color=state,
                    symbol='circle'
                ),
                name=f'Trafik Işığı {edge_id}'
            ))
    
    # Grafik düzenini ayarla
    fig.update_layout(
        title='Trafik Simülasyonu',
        xaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False),
        yaxis=dict(range=[-1.5, 1.5], showgrid=False, zeroline=False),
        showlegend=False,
        height=600,
        width=600
    )
    
    container.plotly_chart(fig, use_container_width=True)

def plot_performance_metrics(metrics_history, container):
    """Performans metriklerini görselleştirir"""
    if not metrics_history:
        return
    
    # Metrikleri numpy dizilerine dönüştür
    waiting_times = np.array([m['waiting_time'] for m in metrics_history])
    queue_lengths = np.array([m['queue_length'] for m in metrics_history])
    average_speeds = np.array([m['average_speed'] for m in metrics_history])
    rewards = np.array([m['reward'] for m in metrics_history])
    
    # Alt grafikler oluştur
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Ortalama Bekleme Süresi',
            'Kuyruk Uzunluğu',
            'Ortalama Hız',
            'Ödül'
        )
    )
    
    # Bekleme süresi grafiği
    fig.add_trace(
        go.Scatter(
            y=waiting_times,
            name='Bekleme Süresi',
            line=dict(color='red')
        ),
        row=1, col=1
    )
    
    # Kuyruk uzunluğu grafiği
    fig.add_trace(
        go.Scatter(
            y=queue_lengths,
            name='Kuyruk Uzunluğu',
            line=dict(color='blue')
        ),
        row=1, col=2
    )
    
    # Ortalama hız grafiği
    fig.add_trace(
        go.Scatter(
            y=average_speeds,
            name='Ortalama Hız',
            line=dict(color='green')
        ),
        row=2, col=1
    )
    
    # Ödül grafiği
    fig.add_trace(
        go.Scatter(
            y=rewards,
            name='Ödül',
            line=dict(color='purple')
        ),
        row=2, col=2
    )
    
    # Grafik düzenini güncelle
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text='Trafik Sinyal Performans Metrikleri'
    )
    
    # Grafikleri göster
    container.plotly_chart(fig, use_container_width=True)

def plot_traffic_state(env, container):
    """Trafik durumunu görselleştirir"""
    # Basit kavşak mı karmaşık mı kontrol et
    if hasattr(env, 'traffic_light_states') and 'E2C' in env.traffic_light_states:
        # Basit kavşak
        traffic_light_states = {
            'E2C': env.traffic_light_states['E2C'],
            'W2C': env.traffic_light_states['W2C'],
            'N2C': env.traffic_light_states['N2C'],
            'S2C': env.traffic_light_states['S2C']
        }
        vehicle_counts = {
            'E2C': env.vehicle_counts['E2C'],
            'W2C': env.vehicle_counts['W2C'],
            'N2C': env.vehicle_counts['N2C'],
            'S2C': env.vehicle_counts['S2C']
        }
        queue_lengths = {
            'E2C': env.queue_lengths['E2C'],
            'W2C': env.queue_lengths['W2C'],
            'N2C': env.queue_lengths['N2C'],
            'S2C': env.queue_lengths['S2C']
        }
        color_map = {
            'red': '🔴',
            'yellow': '🟡',
            'green': '��'
        }
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Yön', 'Trafik Işığı', 'Araç Sayısı', 'Kuyruk Uzunluğu'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=[
                    list(traffic_light_states.keys()),
                    [color_map[state] for state in traffic_light_states.values()],
                    list(vehicle_counts.values()),
                    list(queue_lengths.values())
                ],
                fill_color='lavender',
                align='left'
            )
        )])
        fig.update_layout(
            title='Trafik Durumu',
            height=400
        )
        container.plotly_chart(fig, use_container_width=True)
    else:
        # Karmaşık kavşak
        # Aktif fazı ve hareketleri gösteren tablo
        faz = env.phases[env.phase] if hasattr(env, 'phases') else str(env.phase)
        hareketler = env.phase_movements[env.phase] if hasattr(env, 'phase_movements') else []
        table_data = [
            [m for m in hareketler],
            [env.vehicle_counts[m] for m in hareketler],
            [env.queue_lengths[m] for m in hareketler],
            [env.waiting_times[m] for m in hareketler]
        ]
        fig = go.Figure(data=[go.Table(
            header=dict(
                values=['Hareket', 'Araç Sayısı', 'Kuyruk Uzunluğu', 'Bekleme Süresi'],
                fill_color='paleturquoise',
                align='left'
            ),
            cells=dict(
                values=table_data,
                fill_color='lavender',
                align='left'
            )
        )])
        fig.update_layout(
            title=f'Karmaşık Kavşak - Aktif Faz: {faz}',
            height=400
        )
        container.plotly_chart(fig, use_container_width=True)

def plot_model_comparison(dl_metrics, rl_metrics, container):
    """Model karşılaştırma grafiği"""
    fig = go.Figure()
    
    # Derin öğrenme metrikleri
    fig.add_trace(go.Scatter(
        y=dl_metrics,
        name="Derin Öğrenme",
        line=dict(color='blue')
    ))
    
    # Pekiştirmeli öğrenme metrikleri
    fig.add_trace(go.Scatter(
        y=rl_metrics,
        name="Pekiştirmeli Öğrenme",
        line=dict(color='red')
    ))
    
    # Grafik düzenini ayarla
    fig.update_layout(
        title="Model Karşılaştırması",
        xaxis_title="Adım",
        yaxis_title="Toplam Ödül",
        showlegend=True
    )
    
    # Grafiği göster
    container.plotly_chart(fig, use_container_width=True)

def plot_traffic_heatmap(traffic_data, container):
    """Trafik yoğunluğu ısı haritası"""
    fig = go.Figure(data=go.Heatmap(
        z=traffic_data,
        colorscale='RdYlGn_r',
        showscale=True
    ))
    
    # Grafik düzenini ayarla
    fig.update_layout(
        title="Trafik Yoğunluğu Isı Haritası",
        xaxis_title="Yön",
        yaxis_title="Zaman"
    )
    
    # Grafiği göster
    container.plotly_chart(fig, use_container_width=True)

def plot_complex_intersection(env, container):
    """Karmaşık kavşak için faz, hareket ve kuyrukları gösteren basit şema"""
    fig = go.Figure()
    # Fazı başlıkta göster
    fig.update_layout(title=f'Karmaşık Kavşak - Aktif Faz: {env.phases[env.phase]}', height=500, width=500)
    # Her hareket için ok ve kuyruk uzunluğu
    movement_coords = {
        'N->S': ((0.5, 1.2), (0.5, 0.8)),
        'S->N': ((0.5, -0.2), (0.5, 0.2)),
        'E->W': ((1.2, 0.5), (0.8, 0.5)),
        'W->E': ((-0.2, 0.5), (0.2, 0.5)),
        'N->L': ((0.5, 1.2), (0.2, 0.8)),
        'S->L': ((0.5, -0.2), (0.8, 0.2)),
        'E->L': ((1.2, 0.5), (0.8, 0.8)),
        'W->L': ((-0.2, 0.5), (0.2, 0.2)),
        'N->R': ((0.5, 1.2), (0.8, 0.8)),
        'S->R': ((0.5, -0.2), (0.2, 0.2)),
        'E->R': ((1.2, 0.5), (0.8, 0.2)),
        'W->R': ((-0.2, 0.5), (0.2, 0.8)),
        'E->N': ((1.2, 0.5), (0.8, 0.8)),
        'E->S': ((1.2, 0.5), (0.8, 0.2)),
        'S->E': ((0.5, -0.2), (0.8, 0.5)),
        'N->E': ((0.5, 1.2), (0.8, 0.5)),
    }
    for m, (start, end) in movement_coords.items():
        color = 'green' if m in env.phase_movements[env.phase] else 'gray'
        width = 2 + env.queue_lengths.get(m, 0)  # Kuyruk uzunluğuna göre kalınlık
        fig.add_trace(go.Scatter(
            x=[start[0], end[0]],
            y=[start[1], end[1]],
            mode='lines+markers+text',
            line=dict(color=color, width=width),
            marker=dict(size=8),
            text=[f'{m}', f'Kuyruk: {env.queue_lengths.get(m, 0)}'],
            textposition='top right',
            showlegend=False
        ))
    fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
    container.plotly_chart(fig, use_container_width=True) 