import numpy as np
import random

class ComplexIntersectionEnvironment:
    def __init__(self, simulation_time=1000, traffic_density=0.5):
        self.simulation_time = simulation_time
        self.traffic_density = traffic_density
        self.current_step = 0
        # Fazlar: A, B, C, D, E, F (0-5)
        self.phases = ['A', 'B', 'C', 'D', 'E', 'F']
        self.phase = 0  # Başlangıç fazı
        # Her fazda hangi hareketlere izin var (örn: düz, sağa, sola)
        # Örnek: {'A': ['N->S', 'S->N', 'N->L', 'S->L'], ...}
        self.phase_movements = {
            0: ['N->S', 'S->N', 'N->L', 'S->L'],  # A
            1: ['N->R', 'S->R'],                  # B
            2: ['E->N', 'E->S', 'E->L', 'E->R'], # C
            3: ['E->W', 'W->E', 'E->L', 'W->L'], # D
            4: ['W->R', 'E->R'],                  # E
            5: ['S->E', 'N->E', 'S->L', 'N->L']   # F
        }
        # Her hareket için kuyruk ve araç sayısı
        self.movements = ['N->S', 'S->N', 'N->L', 'S->L', 'N->R', 'S->R',
                         'E->W', 'W->E', 'E->L', 'W->L', 'E->N', 'E->S', 'E->R',
                         'W->R', 'S->E', 'N->E']
        self.vehicle_counts = {m: 0 for m in self.movements}
        self.waiting_times = {m: 0 for m in self.movements}
        self.queue_lengths = {m: 0 for m in self.movements}
        self.avg_speeds = {m: 0 for m in self.movements}
    
    def reset(self):
        self.current_step = 0
        self.phase = 0
        for m in self.movements:
            self.vehicle_counts[m] = 0
            self.waiting_times[m] = 0
            self.queue_lengths[m] = 0
            self.avg_speeds[m] = 0
        return self.get_state()
    
    def get_state(self):
        # Her hareket için: araç sayısı, bekleme süresi, kuyruk, hız
        state = []
        for m in self.movements:
            state.extend([
                self.vehicle_counts[m],
                self.waiting_times[m],
                self.queue_lengths[m],
                self.avg_speeds[m]
            ])
        # Şu anki fazı one-hot olarak ekle
        phase_onehot = [1 if i == self.phase else 0 for i in range(len(self.phases))]
        state.extend(phase_onehot)
        return np.array(state)
    
    def step(self, action):
        # Aksiyon: yeni faz (0-5)
        self.phase = action
        # Araç ekle
        self._add_new_vehicles()
        # Araçları hareket ettir (sadece aktif fazdaki hareketler geçebilir)
        self._move_vehicles()
        # Bekleme sürelerini güncelle
        self._update_waiting_times()
        self.current_step += 1
        next_state = self.get_state()
        reward = self._calculate_reward()
        done = self.current_step >= self.simulation_time
        return next_state, reward, done
    
    def _add_new_vehicles(self):
        for m in self.movements:
            if random.random() < self.traffic_density * 0.05:
                self.vehicle_counts[m] += 1
                self.queue_lengths[m] += 1
    
    def _move_vehicles(self):
        # Sadece aktif fazdaki hareketler geçebilir
        for m in self.phase_movements[self.phase]:
            if self.vehicle_counts[m] > 0:
                self.vehicle_counts[m] -= 1
                self.queue_lengths[m] -= 1
                self.avg_speeds[m] = 13.89  # 50 km/s
    
    def _update_waiting_times(self):
        for m in self.movements:
            if self.vehicle_counts[m] > 0:
                self.waiting_times[m] += 1
    
    def _calculate_reward(self):
        total_waiting = sum(self.waiting_times.values())
        total_queue = sum(self.queue_lengths.values())
        total_speed = sum(self.avg_speeds.values())
        avg_speed = total_speed / len(self.avg_speeds) if len(self.avg_speeds) > 0 else 0
        reward = -0.1 * total_waiting - 0.05 * total_queue + 1.0 * total_speed
        if total_queue == 0:
            reward += 500
        if avg_speed > 12:
            reward += 50
        # Faz değişimi cezası kaldırıldı
        self.last_phase = self.phase
        return reward
    
    def close(self):
        pass 