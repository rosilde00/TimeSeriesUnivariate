Predizione di una serie temporale univariata e multivatiata tramite LSTM, utilizzando il framework Pytorch.

La predizione univariata è stata implementata utilizzando un dataset contenente le informazioni sulle temperature minime registrate a Melbourne dal 1980. L'obbiettivo è predire la temperatura del giorno t + window_size + 1 avendo a disposizione le temperature da t a t + window_size (con una window_size di 365 giorni).

Per la predizione temporale multivariata, invece, si è utilizzato un dataset contenente informazioni sul dispendio energetico di una casa, misurato ogni minuto per quattro anni. Esso contiene diverse grandezze elettriche e alcuni valori di sotto-misurazione. L'obbiettivo è predirre il dispendio energetico attivo al minuto t + window_size +1, avendo tutte le informazioni del dataset dell'intervallo da t a t + window_size (con una window_size di 100 minuti).
