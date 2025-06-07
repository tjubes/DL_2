import torch
import torch.nn as nn

class CNN_LSTM_Net(nn.Module):
    """
    Hybrid CNN + LSTM model for MEG classification.

    Architecture:
    1. CNN layers extract spatial features from MEG sensors.
    2. Reshape CNN output to form a temporal sequence.
    3. LSTM models the sequence over time.
    4. Fully connected classifier makes predictions.
    """

    def __init__(self, num_classes=4, input_sensors=248, input_time_steps=2227,
                 lstm_hidden_size=64, num_lstm_layers=2, dropout=0.3):
        super(CNN_LSTM_Net, self).__init__()

        self.input_sensors = input_sensors
        self.input_time_steps = input_time_steps

        # Step 1: Spatial CNN layers (learn spatial patterns across sensors)
        self.spatial_conv = nn.Sequential(
            # Reduce number of spatial (sensor) rows while keeping time dimension intact
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(16, 1), stride=(4, 1), padding=(6, 0)),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, kernel_size=(8, 1), stride=(2, 1), padding=(3, 0)),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout2d(dropout),

            nn.Conv2d(32, 64, kernel_size=(4, 1), stride=(2, 1), padding=(1, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        # Calculate the total spatial feature dimension after the conv stack
        self._calculate_spatial_output_size()

        # Step 2: Temporal Conv1D layers to process across time
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(self.flattened_spatial_dim, 128, kernel_size=16, stride=8, padding=4),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv1d(128, 64, kernel_size=8, stride=4, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        self._calculate_temporal_output_size()

        # Step 3: LSTM for modeling sequential dynamics over time
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            batch_first=True,
            dropout=dropout if num_lstm_layers > 1 else 0,
            bidirectional=True
        )

        # Step 4: Fully connected classifier
        lstm_output_size = lstm_hidden_size * 2  # bidirectional doubles output
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_size, lstm_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden_size, num_classes)
        )

        print(f"CNNLSTMNet created with {sum(p.numel() for p in self.parameters()):,} parameters")
        print(f"  - Flattened spatial features per time step: {self.flattened_spatial_dim}")
        print(f"  - Temporal sequence length after temporal convs: {self.temporal_length}")
        print(f"  - LSTM output dim per time step: {lstm_output_size}")

    def _calculate_spatial_output_size(self):
        # Simulate input to compute spatial conv output shape
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, self.input_sensors, self.input_time_steps)
            x = self.spatial_conv(dummy_input)
            _, conv_channels, reduced_sensors, _ = x.shape
            self.flattened_spatial_dim = conv_channels * reduced_sensors  # for Conv1D

    def _calculate_temporal_output_size(self):
        """Simulate a forward pass through temporal_conv to compute output sequence length."""
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.flattened_spatial_dim, self.input_time_steps)
            output = self.temporal_conv(dummy_input)
            self.temporal_length = output.shape[2]  # length after Conv1D

    def forward(self, x):
        batch_size = x.size(0)  # x: (B, 1, sensors=248, time_steps)

        # Step 1: Spatial conv
        x = self.spatial_conv(x)  # → (B, conv_channels=64, reduced_sensors, time_steps)

        # Step 2: Flatten spatial dims (channels × sensors) → one feature vector per time step
        x = x.view(batch_size, self.flattened_spatial_dim, -1)  # → (B, flattened_spatial, time_steps)

        # Step 3: Temporal conv
        x = self.temporal_conv(x)  # → (B, 64, temporal_length)

        # Step 4: Prepare for LSTM: transpose to (B, sequence_len, features)
        x = x.transpose(1, 2)  # → (B, temporal_length, 64)

        # Step 5: LSTM
        lstm_out, _ = self.lstm(x)  # → (B, temporal_length, 128)

        # Step 6: Use the final time step (last one) for classification
        x = lstm_out[:, -1, :]  # → (B, 128)

        # Step 7: Classifier
        out = self.classifier(x)  # → (B, num_classes)
        
        return out
