def build_lstm_model(self):
        """Builds a Keras LSTM model if TensorFlow is available."""
        if not HAS_TF:
            return None
        
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(self.seq_len, 1)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(25))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
