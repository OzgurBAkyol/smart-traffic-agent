import numpy as np
import traci

def calculate_metrics(env):
    """Trafik simülasyonu için performans metriklerini hesaplar"""
    # Basit kavşak mı karmaşık mı kontrol et
    if hasattr(env, 'vehicle_counts') and 'E2C' in env.vehicle_counts:
        # Basit kavşak
        waiting_times = [
            env.waiting_times['E2C'],
            env.waiting_times['W2C'],
            env.waiting_times['N2C'],
            env.waiting_times['S2C']
        ]
        avg_waiting_time = np.mean(waiting_times)
        queue_lengths = [
            env.queue_lengths['E2C'],
            env.queue_lengths['W2C'],
            env.queue_lengths['N2C'],
            env.queue_lengths['S2C']
        ]
        avg_queue_length = np.mean(queue_lengths)
        # Ortalama kuyruk uzunluğunu simülasyon boyunca hesapla
        if not hasattr(env, 'queue_length_history'):
            env.queue_length_history = []
        env.queue_length_history.append(avg_queue_length)
        mean_queue_length = np.mean(env.queue_length_history)
        speeds = [
            env.avg_speeds['E2C'], env.avg_speeds['W2C'], env.avg_speeds['N2C'], env.avg_speeds['S2C'],
            env.avg_speeds['C2E'], env.avg_speeds['C2W'], env.avg_speeds['C2N'], env.avg_speeds['C2S']
        ]
        avg_speed = np.mean(speeds)
        reward = env._calculate_reward() if hasattr(env, '_calculate_reward') else 0
        # Ortalama ödül
        if not hasattr(env, 'reward_history'):
            env.reward_history = []
        env.reward_history.append(reward)
        mean_reward = np.mean(env.reward_history)
        return {
            'waiting_time': avg_waiting_time,
            'queue_length': mean_queue_length,
            'average_speed': avg_speed,
            'reward': mean_reward
        }
    elif hasattr(env, 'movements'):
        # Karmaşık kavşak
        waiting_times = [env.waiting_times[m] for m in env.movements]
        avg_waiting_time = np.mean(waiting_times)
        queue_lengths = [env.queue_lengths[m] for m in env.movements]
        avg_queue_length = np.mean(queue_lengths)
        if not hasattr(env, 'queue_length_history'):
            env.queue_length_history = []
        env.queue_length_history.append(avg_queue_length)
        mean_queue_length = np.mean(env.queue_length_history)
        speeds = [env.avg_speeds[m] for m in env.movements]
        avg_speed = np.mean(speeds)
        reward = env._calculate_reward() if hasattr(env, '_calculate_reward') else 0
        if not hasattr(env, 'reward_history'):
            env.reward_history = []
        env.reward_history.append(reward)
        mean_reward = np.mean(env.reward_history)
        return {
            'waiting_time': avg_waiting_time,
            'queue_length': mean_queue_length,
            'average_speed': avg_speed,
            'reward': mean_reward
        }

def calculate_reward(env):
    """Pekiştirmeli öğrenme için ödül fonksiyonu"""
    # Bekleme süresi cezası
    waiting_penalty = -0.1 * np.mean([
        env.waiting_times['E2C'],
        env.waiting_times['W2C'],
        env.waiting_times['N2C'],
        env.waiting_times['S2C']
    ])
    
    # Kuyruk uzunluğu cezası
    queue_penalty = -0.05 * np.mean([
        env.queue_lengths['E2C'],
        env.queue_lengths['W2C'],
        env.queue_lengths['N2C'],
        env.queue_lengths['S2C']
    ])
    
    # Hız ödülü
    speed_reward = 0.2 * np.mean([
        env.avg_speeds['E2C'],
        env.avg_speeds['W2C'],
        env.avg_speeds['N2C'],
        env.avg_speeds['S2C']
    ])
    
    # Toplam ödül
    total_reward = waiting_penalty + queue_penalty + speed_reward
    
    return total_reward

def calculate_efficiency_metrics(env):
    """Verimlilik metriklerini hesapla"""
    metrics = calculate_metrics(env)
    
    efficiency = {
        'throughput': 0,
        'delay': 0,
        'efficiency_score': 0
    }
    
    # Geçiş kapasitesi (throughput)
    efficiency['throughput'] = metrics['vehicle_count'] / env.current_step if env.current_step > 0 else 0
    
    # Gecikme
    efficiency['delay'] = metrics['waiting_time'] / metrics['vehicle_count'] if metrics['vehicle_count'] > 0 else 0
    
    # Verimlilik skoru
    efficiency['efficiency_score'] = (
        efficiency['throughput'] * 0.4 +
        (1 / (1 + efficiency['delay'])) * 0.3 +
        (metrics['average_speed'] / 50) * 0.3  # 50 km/s maksimum hız varsayımı
    )
    
    return efficiency

def compare_models(dl_metrics, rl_metrics):
    """Derin öğrenme ve pekiştirmeli öğrenme modellerinin performanslarını karşılaştırır"""
    comparison = {
        'waiting_time': {
            'dl': np.mean([m['waiting_time'] for m in dl_metrics]),
            'rl': np.mean([m['waiting_time'] for m in rl_metrics])
        },
        'queue_length': {
            'dl': np.mean([m['queue_length'] for m in dl_metrics]),
            'rl': np.mean([m['queue_length'] for m in rl_metrics])
        },
        'average_speed': {
            'dl': np.mean([m['average_speed'] for m in dl_metrics]),
            'rl': np.mean([m['average_speed'] for m in rl_metrics])
        },
        'reward': {
            'dl': np.mean([m['reward'] for m in dl_metrics]),
            'rl': np.mean([m['reward'] for m in rl_metrics])
        }
    }
    
    return comparison 