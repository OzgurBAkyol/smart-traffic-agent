import numpy as np
import random

class TrafficEnvironment:
    def __init__(self, simulation_time=1000, traffic_density=0.5):
        self.simulation_time = simulation_time
        self.traffic_density = traffic_density
        self.current_step = 0
        
        # Trafik ışığı durumu
        self.traffic_light_state = 0  # 0: Doğu-Batı yeşil, 1: Kuzey-Güney yeşil
        
        # Trafik ışığı durumları
        self.traffic_light_states = {
            'E2C': 'red', 'W2C': 'red', 'N2C': 'green', 'S2C': 'green',
            'C2E': 'red', 'C2W': 'red', 'C2N': 'green', 'C2S': 'green'
        }
        
        # Trafik durumu
        self.vehicle_counts = {
            'E2C': 0, 'W2C': 0, 'N2C': 0, 'S2C': 0,
            'C2E': 0, 'C2W': 0, 'C2N': 0, 'C2S': 0
        }
        
        # Ortalama hızlar
        self.avg_speeds = {
            'E2C': 0, 'W2C': 0, 'N2C': 0, 'S2C': 0,
            'C2E': 0, 'C2W': 0, 'C2N': 0, 'C2S': 0
        }
        
        # Bekleme süreleri
        self.waiting_times = {
            'E2C': 0, 'W2C': 0, 'N2C': 0, 'S2C': 0,
            'C2E': 0, 'C2W': 0, 'C2N': 0, 'C2S': 0
        }
        
        # Kuyruk uzunlukları
        self.queue_lengths = {
            'E2C': 0, 'W2C': 0, 'N2C': 0, 'S2C': 0,
            'C2E': 0, 'C2W': 0, 'C2N': 0, 'C2S': 0
        }
        
        # Çıkış buffer'ı (her araç çıkışta 3 adım bekler)
        self.exit_buffer = {'C2E': [], 'C2W': [], 'C2N': [], 'C2S': []}
    
    def reset(self):
        """Simülasyonu sıfırla"""
        self.current_step = 0
        self.traffic_light_state = 0
        
        # Trafik ışığı durumlarını sıfırla
        self.traffic_light_states = {
            'E2C': 'red', 'W2C': 'red', 'N2C': 'green', 'S2C': 'green',
            'C2E': 'red', 'C2W': 'red', 'C2N': 'green', 'C2S': 'green'
        }
        
        # Trafik durumunu sıfırla
        for key in self.vehicle_counts:
            self.vehicle_counts[key] = 0
            self.avg_speeds[key] = 0
            self.waiting_times[key] = 0
            self.queue_lengths[key] = 0
        
        # Çıkış buffer'ını sıfırla
        self.exit_buffer = {'C2E': [], 'C2W': [], 'C2N': [], 'C2S': []}
        
        return self.get_state()
    
    def get_state(self):
        """Mevcut durumu al"""
        state = []
        
        # Her yön için trafik durumunu al
        for edge_id in ['E2C', 'W2C', 'N2C', 'S2C', 'C2E', 'C2W', 'C2N', 'C2S']:
            state.extend([
                self.vehicle_counts[edge_id],
                self.avg_speeds[edge_id],
                self.waiting_times[edge_id],
                self.queue_lengths[edge_id]
            ])
        
        return np.array(state)
    
    def step(self, action):
        """Bir adım ilerle"""
        # Trafik ışığı durumunu güncelle
        self.traffic_light_state = action
        
        # Trafik ışığı durumlarını güncelle
        if action == 0:  # Doğu-Batı yeşil
            self.traffic_light_states = {
                'E2C': 'green', 'W2C': 'green', 'N2C': 'red', 'S2C': 'red',
                'C2E': 'green', 'C2W': 'green', 'C2N': 'red', 'C2S': 'red'
            }
        else:  # Kuzey-Güney yeşil
            self.traffic_light_states = {
                'E2C': 'red', 'W2C': 'red', 'N2C': 'green', 'S2C': 'green',
                'C2E': 'red', 'C2W': 'red', 'C2N': 'green', 'C2S': 'green'
            }
        
        # Yeni araçları ekle
        self._add_new_vehicles()
        
        # Araçları hareket ettir
        self._move_vehicles()
        
        # Bekleme sürelerini güncelle
        self._update_waiting_times()
        
        # Çıkıştaki araçları azalt
        self._remove_exited_vehicles()
        
        # Simülasyon adımını artır
        self.current_step += 1
        
        # Yeni durumu al
        next_state = self.get_state()
        
        # Ödülü hesapla
        reward = self._calculate_reward()
        
        # Bitiş durumunu kontrol et
        done = self.current_step >= self.simulation_time
        
        return next_state, reward, done
    
    def _add_new_vehicles(self):
        """Yeni araçları ekle"""
        for edge_id in ['E2C', 'W2C', 'N2C', 'S2C']:
            if random.random() < self.traffic_density * 0.1:
                self.vehicle_counts[edge_id] += 1
                self.queue_lengths[edge_id] += 1
    
    def _move_vehicles(self):
        """Araçları hareket ettir"""
        # Doğu-Batı yönü
        if self.traffic_light_state == 0:
            # Doğu'dan Batı'ya
            if self.vehicle_counts['E2C'] > 0:
                self.vehicle_counts['E2C'] -= 1
                self.queue_lengths['E2C'] -= 1
                self.vehicle_counts['C2W'] += 1
                self.avg_speeds['C2W'] = 13.89
                self.exit_buffer['C2W'].append(3)  # 3 adım sonra çıkacak
                self.avg_speeds['E2C'] = 13.89  # Girişte de hız güncelle
            
            # Batı'dan Doğu'ya
            if self.vehicle_counts['W2C'] > 0:
                self.vehicle_counts['W2C'] -= 1
                self.queue_lengths['W2C'] -= 1
                self.vehicle_counts['C2E'] += 1
                self.avg_speeds['C2E'] = 13.89
                self.exit_buffer['C2E'].append(3)
                self.avg_speeds['W2C'] = 13.89
        
        # Kuzey-Güney yönü
        else:
            # Kuzey'den Güney'e
            if self.vehicle_counts['N2C'] > 0:
                self.vehicle_counts['N2C'] -= 1
                self.queue_lengths['N2C'] -= 1
                self.vehicle_counts['C2S'] += 1
                self.avg_speeds['C2S'] = 13.89
                self.exit_buffer['C2S'].append(3)
                self.avg_speeds['N2C'] = 13.89
            
            # Güney'den Kuzey'e
            if self.vehicle_counts['S2C'] > 0:
                self.vehicle_counts['S2C'] -= 1
                self.queue_lengths['S2C'] -= 1
                self.vehicle_counts['C2N'] += 1
                self.avg_speeds['C2N'] = 13.89
                self.exit_buffer['C2N'].append(3)
                self.avg_speeds['S2C'] = 13.89
    
    def _update_waiting_times(self):
        """Bekleme sürelerini güncelle"""
        for edge_id in ['E2C', 'W2C', 'N2C', 'S2C']:
            if self.vehicle_counts[edge_id] > 0:
                self.waiting_times[edge_id] += 1
    
    def _calculate_reward(self):
        """
        Faz bazlı ekstra ödül/ceza: Aktif fazdaki toplam kuyruk ve bekleme süresi çok fazlaysa ekstra ceza, azsa ekstra ödül.
        """
        reward = 0
        total_waiting_time = sum(self.waiting_times.values())
        total_queue = sum(self.queue_lengths.values())
        # Kuyruk uzunluğuna maksimum ceza
        reward -= total_queue * 100
        # Bekleme süresine çok büyük ceza
        reward -= total_waiting_time * 10
        # Faz bazlı ceza/ödül
        if hasattr(self, 'traffic_light_state'):
            if self.traffic_light_state == 0:
                # Doğu-Batı fazı
                phase_queues = self.queue_lengths['E2C'] + self.queue_lengths['W2C']
                phase_waits = self.waiting_times['E2C'] + self.waiting_times['W2C']
            else:
                # Kuzey-Güney fazı
                phase_queues = self.queue_lengths['N2C'] + self.queue_lengths['S2C']
                phase_waits = self.waiting_times['N2C'] + self.waiting_times['S2C']
            if phase_queues + phase_waits > 10:
                reward -= 1000  # Fazda çok kuyruk/bekleme varsa ekstra ceza
            elif phase_queues + phase_waits == 0:
                reward += 1000  # Fazda hiç kuyruk yoksa ekstra ödül
        # Hiç kuyruk yoksa devasa ödül
        if total_queue == 0:
            reward += 10000
        # Faz değişimi cezası çok küçük
        if self.current_step > 0 and self.traffic_light_state != self.traffic_light_states.get('last_state', None):
            reward -= 0.01
        self.traffic_light_states['last_state'] = self.traffic_light_state
        return reward
    
    def _remove_exited_vehicles(self):
        """Çıkış yollarındaki araçları sistemden çıkar"""
        for edge_id in ['C2E', 'C2W', 'C2N', 'C2S']:
            if self.exit_buffer[edge_id]:
                self.exit_buffer[edge_id] = [t-1 for t in self.exit_buffer[edge_id]]
                if self.exit_buffer[edge_id][0] <= 0:
                    self.exit_buffer[edge_id].pop(0)
                    if self.vehicle_counts[edge_id] > 0:
                        self.vehicle_counts[edge_id] -= 1
                        self.avg_speeds[edge_id] = 0
    
    def close(self):
        """Simülasyonu kapat"""
        pass 