import torch
import torch.nn as nn

class Enhanced_NN_Reg(nn.Module):
    def __init__(self, nb_features, hidden_dim=128, dropout_rate=0.2):
        super(Enhanced_NN_Reg, self).__init__()

        # Couche d'entrée
        self.input_layer = nn.Sequential(
            nn.Linear(nb_features, hidden_dim),
            nn.SiLU()
        )

        # Bloc Résiduel 1
        self.block1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # Bloc Résiduel 2
        self.block2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate)
        )

        # Couche de sortie
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.SiLU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(32, 1),
            nn.Softplus() # L'IV reste toujours positive
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. On passe l'entrée
        x = self.input_layer(x)

        # 2. Skip Connection 1 : On additionne l'entrée à la sortie du bloc 1
        x = x + self.block1(x)

        # 3. Skip Connection 2 : On additionne à nouveau
        x = x + self.block2(x)

        # 4. Prédiction finale
        return self.output_layer(x)