import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque

# En iyi cihazı seçen fonksiyon

def get_best_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class DeepLearningModel:
    def __init__(self, input_size=32, hidden_size=128, output_size=4, num_phases=2):
        self.device = get_best_device()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
        # Son 10 adım için geçmişi tut
        self.last_scores = deque(maxlen=10)
        self.force_green = None
        self.force_green_steps = 0
        self.num_phases = num_phases  # 2 (basit) veya 6 (karmaşık)
        
        # Modeli değerlendirme moduna al
        self.model.eval()
        
    def predict(self, state):
        with torch.no_grad():  # Gradient hesaplamayı devre dışı bırak
            # Karmaşık kavşak için 6 fazlı aksiyon uzayı
            if self.num_phases == 6:
                # Her faz için toplam kuyruk ve bekleme süresi skorunu hesapla
                phase_scores = []
                for phase in range(6):
                    # State vektöründe her hareket için 4 değer var (araç, bekleme, kuyruk, hız)
                    # 16 hareket * 4 = 64, faz one-hot 6 = toplam 70
                    # Faz hareketleri:
                    phase_movements = [
                        [0,1,2,3],    # A: N->S, S->N, N->L, S->L
                        [4,5],        # B: N->R, S->R
                        [10,11,8,12], # C: E->N, E->S, E->L, E->R
                        [6,7,8,9],    # D: E->W, W->E, E->L, W->L
                        [13,12],      # E: W->R, E->R
                        [14,15,3,2]   # F: S->E, N->E, S->L, N->L
                    ][phase]
                    score = 0
                    for idx in phase_movements:
                        queue = state[idx*4+2]
                        wait = state[idx*4+1]
                        score += 5*queue + 5*wait
                    phase_scores.append(score)
                # En yüksek skora sahip fazı seç
                return int(np.argmax(phase_scores))
            # Basit kavşak için eski kural
            # Eşikler
            QUEUE_THRESHOLD = 5
            WAIT_THRESHOLD = 30
            SPEED_HIGH = 10
            FORCE_GREEN_STEPS = 3
            # Kuyruk, bekleme süresi, hız
            east_west_queue = state[3] + state[7]
            north_south_queue = state[11] + state[15]
            east_west_wait = state[1] + state[5]
            north_south_wait = state[9] + state[13]
            east_west_speed = state[4] + state[8]
            north_south_speed = state[12] + state[16]
            # Zorunlu yeşil: Eğer bir yönde kuyruk çok fazlaysa, o yöne 3 adım üst üste yeşil ver
            if self.force_green_steps > 0:
                self.force_green_steps -= 1
                return self.force_green
            if east_west_queue > QUEUE_THRESHOLD:
                self.force_green = 0
                self.force_green_steps = FORCE_GREEN_STEPS - 1
                return 0
            if north_south_queue > QUEUE_THRESHOLD:
                self.force_green = 1
                self.force_green_steps = FORCE_GREEN_STEPS - 1
                return 1
            # Zorunlu yeşil: Eğer bir yönde bekleme süresi çok fazlaysa
            if east_west_wait > WAIT_THRESHOLD:
                return 0
            if north_south_wait > WAIT_THRESHOLD:
                return 1
            # Akıllı skor: (bekleme süresi * 5 + kuyruk * 4 - hız * 6)
            east_west_score = 4 * east_west_queue + 5 * east_west_wait - 6 * east_west_speed
            north_south_score = 4 * north_south_queue + 5 * north_south_wait - 6 * north_south_speed
            self.last_scores.append((east_west_score, north_south_score))
            ew_avg = np.mean([s[0] for s in self.last_scores])
            ns_avg = np.mean([s[1] for s in self.last_scores])
            # Eğer kuyruk sıfır ve hız yüksekse ekstra öncelik
            if east_west_queue == 0 and east_west_speed > SPEED_HIGH:
                return 0
            if north_south_queue == 0 and north_south_speed > SPEED_HIGH:
                return 1
            if ew_avg >= ns_avg:
                return 0
            else:
                return 1
    
    def train(self, states, targets):
        """Modeli eğitir"""
        states = torch.FloatTensor(states).to(self.device)
        targets = torch.FloatTensor(targets).to(self.device)
        
        self.optimizer.zero_grad()
        predictions = self.model(states)
        loss = self.criterion(predictions, targets)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def save(self, path):
        """Modeli kaydeder"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }, path)
    
    def load(self, path):
        """Kaydedilmiş modeli yükler"""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict']) 