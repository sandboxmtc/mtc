import streamlit as st
import firebase_admin
from firebase_admin import credentials, db
import pandas as pd
import numpy as np
import xgboost as xgb
import altair as alt
from sklearn.preprocessing import StandardScaler
from datetime import datetime
from streamlit_autorefresh import st_autorefresh

# Cache persistente para armazenar previsões
@st.cache_data(show_spinner=False)
def get_prediction_cache():
    if "prediction_cache" not in st.session_state:
        st.session_state["prediction_cache"] = {}
    return st.session_state["prediction_cache"]

# Inicialização do Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate({
        "type": st.secrets["firebase"]["type"],
        "project_id": st.secrets["firebase"]["project_id"],
        "private_key_id": st.secrets["firebase"]["private_key_id"],
        "private_key": st.secrets["firebase"]["private_key"],
        "client_email": st.secrets["firebase"]["client_email"],
        "client_id": st.secrets["firebase"]["client_id"],
        "auth_uri": st.secrets["firebase"]["auth_uri"],
        "token_uri": st.secrets["firebase"]["token_uri"],
        "auth_provider_x509_cert_url": st.secrets["firebase"]["auth_provider_x509_cert_url"],
        "client_x509_cert_url": st.secrets["firebase"]["client_x509_cert_url"],
        "universe_domain": st.secrets["firebase"]["universe_domain"]
    })
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://crazytime-pedro-2022-default-rtdb.firebaseio.com/'
    })

# Referência ao nó do banco de dados
ref = db.reference("/Double_blaze2")

# Carregar o modelo XGBoost
model = xgb.Booster()
model.load_model("modelo_xgboost.json")

def ensure_chronological_predictions(df, cache):
    """
    Garante que as previsões sejam feitas em ordem cronológica,
    evitando recalcular previsões já armazenadas.
    """
    if 'when' in df.columns:
        df = df[df['when'].notna()]  # Remove valores NaN
        df['when'] = df['when'].astype(str)  # Converte para string antes de tentar datetime
        df['when'] = pd.to_datetime(df['when'], errors='coerce')  # Converte, ignorando erros
        df = df.dropna(subset=['when'])  # Remove linhas que não puderam ser convertidas
        df = df.sort_values('when')

    # Garantir que todas as chaves do cache são strings e não estão vazias
    valid_keys = [k for k in cache.keys() if isinstance(k, str) and k.strip()]

    # Converter apenas chaves válidas para datetime
    cached_timestamps = set(pd.to_datetime(valid_keys, errors='coerce').dropna())

    # Filtrar apenas registros novos
    new_records = df[~df['when'].isin(cached_timestamps)]

    return new_records

def calculate_future_whites_multi_window(df):
    """
    Calcula o número de brancos para diferentes janelas temporais: 10, 20 e 30 eventos
    """
    # Converte timestamp para datetime se ainda não for
    if df['timestamp'].dtype == 'object':
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Ordena do mais antigo para o mais novo
    df = df.sort_values('timestamp', ascending=True).reset_index(drop=True)
    
    whites_10 = []
    whites_20 = []
    whites_30 = []
    
    for i in range(len(df)):
        # Para próximos 10 eventos
        if i + 10 >= len(df):
            whites_10.append(np.nan)
        else:
            next_10_colors = df['color'].iloc[i+1:i+11].tolist()
            whites_10.append(sum(1 for color in next_10_colors if color == 'White'))
        
        # Para próximos 20 eventos
        if i + 20 >= len(df):
            whites_20.append(np.nan)
        else:
            next_20_colors = df['color'].iloc[i+1:i+21].tolist()
            whites_20.append(sum(1 for color in next_20_colors if color == 'White'))
        
        # Para próximos 30 eventos
        if i + 30 >= len(df):
            whites_30.append(np.nan)
        else:
            next_30_colors = df['color'].iloc[i+1:i+31].tolist()
            whites_30.append(sum(1 for color in next_30_colors if color == 'White'))
    
    return whites_10, whites_20, whites_30

def calculate_rounds_until_white(df):
    """
    Calcula quantas rodadas foram necessárias até encontrar um branco
    após cada sinal de predição positivo
    """
    rounds_until_white = []
    
    # Garantir que os dados estão ordenados por timestamp
    df = df.sort_values('timestamp')
    
    for idx in df[df['Predição'] == 1].index:
        # Pegar todas as cores após este índice
        future_colors = df.loc[idx:, 'color'].values
        
        # Procurar o próximo branco
        found_white = False
        for i, color in enumerate(future_colors):
            if color == 'White':
                rounds_until_white.append(i)
                found_white = True
                break
        
        # Se não encontrou branco, marca como None
        if not found_white:
            rounds_until_white.append(None)
    
    return rounds_until_white

def preprocess_data(df):
    # Reset index e renomear coluna
    df = df.reset_index().rename(columns={'id': 'timestamp'})
    
    # Converter colunas categóricas
    df = pd.get_dummies(data=df, columns=['type', 'color'])
    
    # Converter colunas numéricas
    df['payout'] = pd.to_numeric(df['payout'], errors='coerce')
    df['total_bet'] = pd.to_numeric(df['total_bet'], errors='coerce')
    
    # Remover colunas desnecessárias
    df = df.drop(columns=['multiplier', 'spin_result', 'type_Special Result'], errors='ignore')
    
    # Calcular jogadores online
    df['online_players'] = (df['total_bet'] / 17.66).round().astype(int)
    
    # Feature engineering mantendo as mesmas features do modelo original
    rolling_sum_window = 25
    df['bank_profit'] = df['total_bet'] - df['payout']
    df['rolling_sum'] = df['bank_profit'].rolling(window=rolling_sum_window).sum().shift(1)
    
    window_size = 5
    df['open'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[0], raw=True).shift(1)
    df['close'] = df['rolling_sum'].rolling(window=window_size).apply(lambda x: x[-1], raw=True).shift(1)
    df['high'] = df['rolling_sum'].rolling(window=window_size).max().shift(1)
    df['low'] = df['rolling_sum'].rolling(window=window_size).min().shift(1)
    df['velocity'] = df['rolling_sum'].diff().fillna(0)
    df['acceleration'] = df['velocity'].diff(3).fillna(0)
    
    # Bollinger Bands
    dev_window = 10
    mult_bb = 2.0
    df['basis'] = df['close'].rolling(window=dev_window).mean().shift(1)
    df['dev'] = df['close'].rolling(window=dev_window).std().shift(1)
    df['upperBB'] = df['basis'] + mult_bb * df['dev']
    df['lowerBB'] = df['basis'] - mult_bb * df['dev']
    
    # MACD
    short_window_macd = 7
    long_window_macd = 25
    signal_window = 11
    df['ema_short'] = df['close'].ewm(span=short_window_macd, min_periods=1, adjust=False).mean().shift(1)
    df['ema_long'] = df['close'].ewm(span=long_window_macd, min_periods=1, adjust=False).mean().shift(1)
    df['MACD'] = df['ema_short'] - df['ema_long']
    df['Signal_Line'] = df['MACD'].ewm(span=signal_window, min_periods=1, adjust=False).mean().shift(1)
    df['MACD_Histogram'] = df['MACD'] - df['Signal_Line']
    df['MACD_Histogram_Signal_Interaction'] = df['MACD_Histogram'] * df['Signal_Line']
    
    # Moving Averages
    ma_long_window = 40
    slope_ma_long_window = 30
    df['MA_short'] = df['bank_profit'].rolling(window=10, min_periods=1).mean().shift(1)
    df['MA_long'] = df['bank_profit'].rolling(window=ma_long_window, min_periods=1).mean().shift(1)
    df['Slope_MA_long'] = df['MA_long'].diff().shift(1) / slope_ma_long_window
    df['MA_longer'] = df['bank_profit'].rolling(window=120, min_periods=1).mean().shift(1)
    
    # Normalização
    scaler = StandardScaler()
    numerical_features = [
        'dev', 'total_bet', 'online_players', 'MACD', 'MACD_Histogram',
        'MA_short', 'MA_long', 'MA_longer', 'Signal_Line', 'acceleration', 'Slope_MA_long'
    ]
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    # Garantir todas as features necessárias
    expected_features = model.feature_names
    for col in expected_features:
        if col not in df.columns:
            df[col] = 0
    
    df = df[expected_features]
    return df

def predict_with_model(df):
    features_for_model = df.drop(columns=['when', 'color'], errors='ignore')
    dmatrix = xgb.DMatrix(features_for_model)
    predictions = model.predict(dmatrix)
    return predictions

# Interface no Streamlit
st.title("Shark MTC - Caça Brancos")
st.write("O app atualizará automaticamente a cada novo resultado. Você também pode clicar manualmente para consultar e prever.")

# # Obter a hora atual do servidor
# current_hour = datetime.now().hour

  
if "last_data" not in st.session_state:
    st.session_state["last_data"] = None
    
refresh_count = st_autorefresh(interval=15000, key="refresh_app", limit=None)

# Rodamos o processamento se o botão for clicado OU se o auto-refresh (refresh_count > 0) ocorrer
if st.button("Consultar e Prever") or refresh_count > 0:
    global last_data
    cache = get_prediction_cache()  # Obtém o cache existente
    
    # Buscar dados do Firebase
    data = ref.order_by_key().limit_to_last(2000).get()
    
    if not data:
        st.error("Nenhum dado encontrado no Firebase!")
    else:
        
        df = pd.DataFrame.from_dict(data, orient='index')

        if 'when' in df.columns:
            df = df[df['when'].notna()]
            df['when'] = df['when'].astype(str)
            df['when'] = pd.to_datetime(df['when'], errors='coerce')
            df = df.dropna(subset=['when'])
                
            # if st.session_state["last_data"] is None or df['when'].iloc[-1] != st.session_state["last_data"]:
                # st.session_state["last_data"] = df['when'].iloc[-1]

            # Filtrar apenas os registros que ainda não foram processados
            new_data = ensure_chronological_predictions(df, cache)
    
            if not new_data.empty:
                processed_data = preprocess_data(new_data)
                predictions = predict_with_model(processed_data)
    
                # Atualizar cache apenas com novos dados
                for i, (index, row) in enumerate(new_data.iterrows()):
                    timestamp_key = str(row['when'])
                    if timestamp_key not in cache:
                        cache[timestamp_key] = {
                            "timestamp": row['when'],
                            "color": row.get('color', None),
                            "Probabilidade": float(predictions[i]),
                            "Predição": int(predictions[i] > 0.8)
                        }
    
            # Apenas mostrar uma mensagem sem exibir a tabela
            st.success("Novas previsões adicionadas ao cache.")
    
            # Converter cache para DataFrame e exibir os resultados
            result_df = pd.DataFrame.from_dict(cache, orient='index')
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            result_df = result_df.sort_values('timestamp', ascending=True)
            
            # Convertemos e ordenamos por timestamp (mais antigo primeiro)
            result_df['timestamp'] = pd.to_datetime(result_df['timestamp'])
            result_df = result_df.sort_values('timestamp', ascending=True)
            
            # Limitar para 1200-1950 linhas para precisão
            limited_df = result_df.iloc[1200:1950].copy()
            limited_df['color_binary'] = (limited_df['color'] == 'White').astype(int)
            
            # Calcular brancos para diferentes janelas temporais
            whites_10, whites_20, whites_30 = calculate_future_whites_multi_window(limited_df)
            
            limited_df['White Count (Next 10)'] = whites_10
            limited_df['White Count (Next 20)'] = whites_20
            limited_df['White Count (Next 30)'] = whites_30
            
            # Calcular True Entry baseado nas três condições
            limited_df['True Entry (10)'] = limited_df['White Count (Next 10)'] >= 1
            limited_df['True Entry (20)'] = limited_df['White Count (Next 20)'] >= 2
            limited_df['True Entry (30)'] = limited_df['White Count (Next 30)'] >= 3
            
            # True Entry combinado (qualquer uma das condições)
            limited_df['True Entry Combined'] = (
                limited_df['True Entry (10)'] |
                limited_df['True Entry (20)'] |
                limited_df['True Entry (30)']
            )
            
            # Cálculo de precisão para entrada combinada (original)
            mask_valid_targets = (
                limited_df['White Count (Next 10)'].notna() &
                limited_df['White Count (Next 20)'].notna() &
                limited_df['White Count (Next 30)'].notna()
            )
            
            true_positives = (limited_df.loc[mask_valid_targets & (limited_df['Predição'] == 1), 'True Entry Combined']).sum()
            predicted_positives = limited_df.loc[mask_valid_targets, 'Predição'].sum()
            precision = (true_positives / predicted_positives) if predicted_positives > 0 else 0
    
            # Nova métrica - 1 branco nos próximos 20
            limited_df['True Entry Single White 20'] = limited_df['White Count (Next 20)'] >= 1
            true_positives_single = (limited_df.loc[mask_valid_targets & (limited_df['Predição'] == 1), 'True Entry Single White 20']).sum()
            precision_single_white = (true_positives_single / predicted_positives) if predicted_positives > 0 else 0
    
            # Cálculo das novas métricas de precisão
            limited_df['True Entry Single White 25'] = limited_df['White Count (Next 20)'] >= 1
            limited_df['True Entry Single White 30'] = limited_df['White Count (Next 30)'] >= 1
            
            true_positives_25 = (limited_df.loc[mask_valid_targets & (limited_df['Predição'] == 1), 'True Entry Single White 25']).sum()
            true_positives_30 = (limited_df.loc[mask_valid_targets & (limited_df['Predição'] == 1), 'True Entry Single White 30']).sum()
            
            precision_single_white_25 = (true_positives_25 / predicted_positives) if predicted_positives > 0 else 0
            precision_single_white_30 = (true_positives_30 / predicted_positives) if predicted_positives > 0 else 0
            
            # Calcular rodadas até o branco
            rounds_until_white = calculate_rounds_until_white(limited_df)
            rounds_df = pd.DataFrame({
                'rounds_until_white': rounds_until_white
            }).dropna()
            
            avg_rounds = rounds_df['rounds_until_white'].mean()
            median_rounds = rounds_df['rounds_until_white'].median()
            
            # Mostrar métricas - Reorganizado para incluir novas métricas
            st.write("Estatísticas:")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total de Previsões Classe 1", f"{predicted_positives}")
                st.metric("Precisão (Condições Combinadas)", f"{precision:.2%}")
                
            with col2:
                st.metric("Precisão (1+ em 20)", f"{precision_single_white:.2%}")
                st.metric("Precisão (1+ em 25)", f"{precision_single_white_25:.2%}")
                
            with col3:
                st.metric("Precisão (1+ em 30)", f"{precision_single_white_30:.2%}")
                st.metric("Média de Rodadas até Branco", f"{avg_rounds:.1f}")
            
            # Gráfico de distribuição de rodadas até o branco
            rounds_hist = alt.Chart(rounds_df).mark_bar().encode(
                x=alt.X('rounds_until_white:Q', 
                       bin=alt.Bin(maxbins=20), 
                       title='Número de Rodadas'),
                y=alt.Y('count():Q', title='Frequência')
            ).properties(
                title='Rodadas acumuladas até o Branco e após sinal verde'
            )
            
            st.altair_chart(rounds_hist, use_container_width=True)
            
            # Estatísticas 
            st.write("Estatísticas de rodadas acumuladas após sinal verde:")
            stats_df = pd.DataFrame({
                'Métrica': ['Média', 'Mediana', 'Mínimo', 'Máximo', 'Desvio Padrão'],
                'Valor': [
                    f"{rounds_df['rounds_until_white'].mean():.1f}",
                    f"{rounds_df['rounds_until_white'].median():.1f}",
                    f"{rounds_df['rounds_until_white'].min():.0f}",
                    f"{rounds_df['rounds_until_white'].max():.0f}",
                    f"{rounds_df['rounds_until_white'].std():.1f}"
                ]
            })
            st.dataframe(stats_df)
            
            # Mostrar detalhamento por janela temporal
            st.write("Detalhamento por Janela Temporal (nas previsões positivas):")
            positives_mask = mask_valid_targets & (limited_df['Predição'] == 1)
            st.write(f"1+ em 10 jogadas: {limited_df.loc[positives_mask, 'True Entry (10)'].mean():.2%}")
            st.write(f"2+ em 20 jogadas: {limited_df.loc[positives_mask, 'True Entry (20)'].mean():.2%}")
            st.write(f"3+ em 30 jogadas: {limited_df.loc[positives_mask, 'True Entry (30)'].mean():.2%}")
            st.write(f"1+ em 20 jogadas: {limited_df.loc[positives_mask, 'True Entry Single White 20'].mean():.2%}")
            
            # Gráfico de precisão por faixa de probabilidade
            bins = [0.8, 0.85, 0.9, 0.95, 1.0]
            labels = ["0.8-0.85", "0.85-0.9", "0.9-0.95", "0.95-1.0"]
            
            limited_df['probability_range'] = pd.cut(
                limited_df['Probabilidade'],
                bins=bins,
                labels=labels,
                right=False
            )
            
            ranges_df = (
                limited_df[mask_valid_targets & (limited_df['Predição'] == 1)]
                .groupby('probability_range')['True Entry Combined']
                .agg(['sum', 'size'])
                .reset_index()
            )
            
            ranges_df.columns = ['probability_range', 'acertos', 'total']
            ranges_df['precisao'] = ranges_df['acertos'] / ranges_df['total']
            
            chart = alt.Chart(ranges_df).mark_bar().encode(
                x=alt.X('probability_range', title="Faixa de Probabilidade"),
                y=alt.Y('precisao', title="Precisão"),
                tooltip=['acertos', 'total', 'precisao']
            ).properties(
                title="Precisão por Faixa de Probabilidade (Condições Combinadas)"
            )
            
            st.altair_chart(chart, use_container_width=True)
            
            # Estilizar tabela
            def highlight_row(row):
                styles = [''] * len(row)
                if row['Predição'] == 1:  # Linhas previstas como classe 1
                    styles = ['background-color: darkgreen; color: white'] * len(row)
                if row['color'] == 'White':  # Linhas onde "White" ocorreu
                    styles = ['background-color: white; color: black'] * len(row)
                return styles
            
            # Usar apenas as colunas que existem no result_df
            columns_to_display = ['timestamp', 'color', 'Predição', 'Probabilidade']
            
            # Criar um DataFrame separado para visualização
            display_df = result_df[columns_to_display].copy()
            
            # Ordenar do mais recente para o mais antigo
            display_df = display_df.sort_values('timestamp', ascending=False)
            
            styled_df = display_df.style.apply(highlight_row, axis=1)
            
            st.write("Tabela de Resultados com Estilização (Mais recentes primeiro):")
            st.write(styled_df)
        # else:
        #     pass

# import os
# import sys
# import psutil

# if __name__ == "__main__":
#     script_path = os.path.abspath(sys.argv[0])
#     if script_path.endswith(".py"):
#         os.system(f'streamlit run "{script_path}" --server.port=8501')
#     else:
#         print("Erro: O nome do arquivo não possui a extensão .py")
