import torch
import torch.nn as nn

class PINNModel(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=128, output_dim=8):
        super(PINNModel, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        self.output_names = [
            "eta_CO2_des",
            "SEEC",
            "CO2_out (%)",
            "Voltage (V)",
            "pH1",
            "Conductivity (mS/cm)",
            "Temp (°C)",
            "Solvent Flow (ml/min)"
        ]

    def forward(self, x):
        return self.net(x)

    def predict_named(self, x):
        out = self.forward(x)
        return {name: out[:, i] for i, name in enumerate(self.output_names)}
