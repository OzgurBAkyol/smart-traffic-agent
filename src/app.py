import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
from dotenv import load_dotenv
import time

# Proje modüllerini içe aktar
from models.deep_learning import DeepLearningModel
from models.reinforcement_learning import RLAgent
from simulation.sumo_env import TrafficEnvironment
from simulation.complex_intersection_env import ComplexIntersectionEnvironment
from utils.visualization import plot_performance_metrics, plot_traffic_state, plot_traffic_simulation, plot_complex_intersection
from utils.metrics import calculate_metrics

# Çevre değişkenlerini yükle
load_dotenv()

st.set_page_config(
    page_title="Akıllı Trafik Sinyal Optimizasyonu",
    page_icon="🚦",
    layout="wide"
)

# Session state'i başlat
if 'simulation_state' not in st.session_state:
    st.session_state.simulation_state = {
        'running': False,
        'current_step': 0,
        'metrics_history': [],
        'env': None,
        'model': None,
        'simulation_completed': False,
        'final_metrics': None,
        'intersection_type': None,
        'model_type': None,
        'last_model_type': None,  # Son seçilen model tipini takip etmek için
        'simulation_speed': 1.0    # Simülasyon hızı çarpanı (1.0 = normal hız)
    }

def initialize_simulation(intersection_type, model_type, simulation_time, traffic_density):
    """Simülasyonu başlatır"""
    if intersection_type == "Karmaşık Kavşak":
        num_phases = 6
        env = ComplexIntersectionEnvironment(
            simulation_time=simulation_time,
            traffic_density=traffic_density
        )
    else:
        num_phases = 2
        env = TrafficEnvironment(
            simulation_time=simulation_time,
            traffic_density=traffic_density
        )
    
    state = env.reset()
    if model_type == "Derin Öğrenme":
        model = DeepLearningModel(num_phases=num_phases)
    else:
        model = RLAgent(input_size=len(state), num_phases=num_phases)
    
    return env, model

def run_simulation_step():
    """Tek bir simülasyon adımını çalıştırır"""
    if not st.session_state.simulation_state['running']:
        return
    
    env = st.session_state.simulation_state['env']
    model = st.session_state.simulation_state['model']
    state = env.get_state()
    
    # Aksiyon seç
    if isinstance(model, DeepLearningModel):
        action = model.predict(state)
    else:
        action = model.select_action(state)
    
    # Ortamı güncelle
    next_state, reward, done = env.step(action)
    
    # Pekiştirmeli öğrenme için deneyimi kaydet
    if isinstance(model, RLAgent):
        model.remember(state, action, reward, next_state, done)
        # Hafif replay - her 10 adımda bir
        if st.session_state.simulation_state['current_step'] % 10 == 0:
            model.replay()
    
    # Metrikleri hesapla
    metrics = calculate_metrics(env)
    st.session_state.simulation_state['metrics_history'].append(metrics)
    
    st.session_state.simulation_state['current_step'] += 1
    
    if done or st.session_state.simulation_state['current_step'] >= env.simulation_time:
        st.session_state.simulation_state['running'] = False
        st.session_state.simulation_state['simulation_completed'] = True
        st.session_state.simulation_state['final_metrics'] = metrics
        save_results(st.session_state.simulation_state['metrics_history'], 
                    st.session_state.simulation_state['model_type'],
                    st.session_state.simulation_state['intersection_type'])

def main():
    st.title("🚦 Akıllı Trafik Sinyal Optimizasyon Sistemi")
    
    # Sidebar
    st.sidebar.header("Simülasyon Ayarları")
    
    # Kavşak tipi seçimi
    intersection_type = st.sidebar.radio(
        "Kavşak Tipi",
        ["Basit Kavşak", "Karmaşık Kavşak"]
    )
    
    # Model seçimi
    model_type = st.sidebar.radio(
        "Model Tipi",
        ["Derin Öğrenme", "Pekiştirmeli Öğrenme"]
    )
    
    # Model değişikliğini kontrol et
    if model_type != st.session_state.simulation_state['last_model_type']:
        st.session_state.simulation_state['running'] = False
        st.session_state.simulation_state['simulation_completed'] = False
        st.session_state.simulation_state['metrics_history'] = []
        st.session_state.simulation_state['last_model_type'] = model_type
        st.rerun()
    
    # Simülasyon parametreleri
    simulation_time = st.sidebar.slider("Simülasyon Süresi (saniye)", 100, 3600, 1000)
    traffic_density = st.sidebar.slider("Trafik Yoğunluğu", 0.1, 1.0, 0.3)
    
    # Simülasyon hızı ayarı
    simulation_speed = st.sidebar.slider(
        "Simülasyon Hızı",
        min_value=0.1,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="1.0 = normal hız, 2.0 = 2x hız, 0.5 = yarım hız"
    )
    st.session_state.simulation_state['simulation_speed'] = simulation_speed
    
    # Progress bar ve yüzde metni en üstte
    progress_bar = st.progress(0)
    progress_text = st.empty()
    
    # Ana içerik
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Trafik Simülasyonu")
        simulation_chart = st.empty()
        
        st.subheader("Trafik Durumu")
        state_chart = st.empty()
    
    with col2:
        st.subheader("Performans Metrikleri")
        performance_chart = st.empty()
        
        st.subheader("Model Metrikleri")
        metrics_container = st.empty()
    
    # Simülasyon kontrolleri
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Simülasyonu Başlat"):
            if not st.session_state.simulation_state['running']:
                st.session_state.simulation_state['env'], st.session_state.simulation_state['model'] = initialize_simulation(
                    intersection_type, model_type, simulation_time, traffic_density
                )
                st.session_state.simulation_state['running'] = True
                st.session_state.simulation_state['current_step'] = 0
                st.session_state.simulation_state['metrics_history'] = []
                st.session_state.simulation_state['simulation_completed'] = False
                st.session_state.simulation_state['intersection_type'] = intersection_type
                st.session_state.simulation_state['model_type'] = model_type
                st.session_state.simulation_state['last_model_type'] = model_type
                st.info(f"Kullanılan cihaz: {st.session_state.simulation_state['model'].device}")
    
    with col2:
        if st.button("Simülasyonu Durdur"):
            st.session_state.simulation_state['running'] = False
            st.session_state.simulation_state['simulation_completed'] = True
    
    # Simülasyon döngüsü veya sonuçları göster
    if st.session_state.simulation_state['running']:
        run_simulation_step()
        
        # Görselleştirmeleri güncelle
        env = st.session_state.simulation_state['env']
        if isinstance(env, ComplexIntersectionEnvironment):
            plot_complex_intersection(env, simulation_chart)
        else:
            plot_traffic_simulation(env, simulation_chart)
        plot_traffic_state(env, state_chart)
        plot_performance_metrics(st.session_state.simulation_state['metrics_history'], performance_chart)
        
        # İlerleme çubuğu ve yüzde metni
        percent = int(100 * st.session_state.simulation_state['current_step'] / simulation_time)
        progress_bar.progress(percent)
        progress_text.text(f"Simülasyonun %{percent} tamamlandı")
        
        # Final metriklerini göster
        if st.session_state.simulation_state['metrics_history']:
            metrics_container.write(st.session_state.simulation_state['metrics_history'][-1])
        
        # Streamlit'in yeniden çizimini tetikle
        # Simülasyon hızına göre bekleme süresi
        time.sleep(0.1 / st.session_state.simulation_state['simulation_speed'])
        st.rerun()
    
    # Simülasyon tamamlandıysa sonuçları göster
    elif st.session_state.simulation_state['simulation_completed']:
        st.success("Simülasyon tamamlandı!")
        
        # Son metrikleri göster
        if st.session_state.simulation_state['metrics_history']:
            plot_performance_metrics(st.session_state.simulation_state['metrics_history'], performance_chart)
            metrics_container.write(st.session_state.simulation_state['metrics_history'][-1])
            
            # Son durumu göster
            env = st.session_state.simulation_state['env']
            if isinstance(env, ComplexIntersectionEnvironment):
                plot_complex_intersection(env, simulation_chart)
            else:
                plot_traffic_simulation(env, simulation_chart)
            plot_traffic_state(env, state_chart)
            
            # İlerleme çubuğunu tamamlandı olarak işaretle
            progress_bar.progress(100)
            progress_text.text("Simülasyon tamamlandı!")

def save_results(metrics_history, model_type, intersection_type):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "data/results"
    os.makedirs(results_dir, exist_ok=True)
    
    filename = f"{results_dir}/{intersection_type}_{model_type}_{timestamp}.npy"
    np.save(filename, metrics_history)
    st.success(f"Sonuçlar kaydedildi: {filename}")

if __name__ == "__main__":
    main() 