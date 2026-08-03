# 📈 Stock Price Prediction — LSTM · RNN · GRU (Animated 3D)

An interactive Streamlit app that trains LSTM, RNN and GRU sequence models
per stock symbol on real NSE daily data (2016–2017) and visualizes the
results with animated 3D charts.

- **LSTM** → predicts next-day **Open** price
- **RNN** → predicts next-day **Close** price
- **GRU** → predicts next-day **Close** price (compared against RNN)

---

## 1. Run it locally

```bash
# 1. Clone / unzip this folder, then cd into it
cd stock-app

# 2. (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the app
streamlit run app.py
```

Streamlit will open automatically at **http://localhost:8501**.
Pick a stock symbol in the sidebar, click **🚀 Train Models**, and watch the
animated 3D chart populate once training finishes (a few seconds per model
since each symbol only has ~400–500 rows).

---

## 2. Deploy it online for free (Streamlit Community Cloud)

1. **Create a GitHub repo** and push this folder to it:
   ```bash
   git init
   git add .
   git commit -m "Stock price prediction app"
   git branch -M main
   git remote add origin https://github.com/<your-username>/<repo-name>.git
   git push -u origin main
   ```
2. Go to **https://share.streamlit.io** and sign in with GitHub.
3. Click **"New app"**, select your repo/branch, and set the main file
   path to `app.py`.
4. Click **Deploy**. The first build takes 2–5 minutes to install
   TensorFlow. After that you'll get a public URL like:
   `https://<your-app-name>.streamlit.app`

That's it — free hosting, HTTPS, and auto-redeploys on every `git push`.

### Alternative free platforms
- **Hugging Face Spaces** (choose the "Streamlit" SDK when creating a Space)
- **Render.com** (free web service, use `streamlit run app.py --server.port $PORT --server.address 0.0.0.0` as the start command)

---

## 3. Project structure

```
stock-app/
├── app.py                  # Streamlit app (UI + models + 3D visuals)
├── requirements.txt
├── Dataset/
│   └── stock_data.csv      # Cleaned NSE daily data (SYMBOL, TIMESTAMP, OPEN, CLOSE)
└── README.md
```

---

## 4. How the models work (plain-English)

- **RNN (Simple Recurrent Neural Network)** reads the sequence of prices
  day by day, carrying forward a "memory" of what it has seen. It tends to
  forget older information quickly.
- **LSTM (Long Short-Term Memory)** adds gates that decide what to keep,
  forget, or output at each step — much better at remembering long-term
  patterns.
- **GRU (Gated Recurrent Unit)** simplifies LSTM's gates, training faster
  while staying close in accuracy.

The app looks back over the last **N days** (adjustable in the sidebar) of
scaled Open/Close prices to predict the next day's price, then plots
**Actual vs Predicted** for the held-out test period.

## 5. Notes on the dataset

The bundled `stock_data.csv` is a cleaned subset (SYMBOL, TIMESTAMP, OPEN,
CLOSE, `SERIES == 'EQ'` only) of NSE bhavcopy data from 2016–2017, so each
symbol forms a genuine daily time series (rather than mixing different
share classes on the same date, which the original raw file contained).
You can also upload your own CSV with the same four columns from the
sidebar.
