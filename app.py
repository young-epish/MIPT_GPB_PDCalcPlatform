from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import sklearn
import pickle


# Для WoE
def splitter(x, col_bondaries):
 
    for i in range(len(col_bondaries)):
        if (i > 0) and (x > col_bondaries[i-1]) and (x < col_bondaries[i]):
                return i
 
        if (i == 0) and (x <= col_bondaries[i]):
                return i
 
        if (i == len(col_bondaries) - 1) and (x > col_bondaries[i]):
                return i+1
        
def score_to_pd(score):
    '''
    Возвращает скор, преобразованный в pd
    '''
    pd_value = 1 / (1 + np.exp(-score))
    return pd_value

def logit_pd(pd_value):
    '''
    Возвращает колонку/лист/одномерный массив с логистически преобразованным PD
    '''
    logit = np.log(pd_value / (1 - pd_value))
    return logit


# Загрузка моделей и объекта масштабирования
# Сохраненная модель модуля Право.ru
with open('models/model_pr.pkl', 'rb') as f:
    model_pr = pickle.load(f)
# Коэффициенты модели модуля СПАРК-Интерфакс`
model_sp = {
    'intercept': -5.146401467044442,
    'coefs': [-0.63500042, -0.52792655, -0.69974848, -0.73613015, -0.53259876]
}
# Коэффициенты для интегральной модели
model_integr = {
    'intercept': 0.7060623578210927,
    'coefs': [0.19324893, 0.98244]
}

with open('models/standard_scaler_sp', 'rb') as f:
    scaler_sp = pickle.load(f)

with open('models/woe_dictionary_sp', 'rb') as f:
    woe_dict = pickle.load(f)

with open('models/woe_boundary_sp', 'rb') as f:
    bondaries = pickle.load(f)

# Список параметров
features_spark = scaler_sp.feature_names_in_.tolist()
# [
#     'p1500', 'p1250_std', 
#     'p2400_a', 'p1700_rt', 
#     'kz_days_rt',
# ]
features_pravo = model_pr.feature_names_
# [
#     'Ratio_ClaimSum_to_Assets', 'Ratio_ClaimSum_to_Revenue', 
#     'Ratio_ClaimSum_to_Capital', 'Ratio_ClaimSum_to_Funds',
#     'Ratio_ClaimSum_to_EBT',
# ]

app = Flask(__name__)


@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route("/predict", methods = ["POST"])
def predict():
    # Входной вектор данных
    input_vector = [float(x) for x in request.form.values()]

    df = pd.DataFrame(input_vector, index=features_spark+features_pravo).T

    # Этот параметр надо прологарифмировать два раза из-за опечатки
    df['p1500'] = np.log(np.log(df['p1500']))

    # Нормализация там, где необходимо
    df[features_spark] = scaler_sp.transform(df[features_spark])

    # WoE
    df_woe_vct = pd.DataFrame()
    for col in features_spark:
        df_woe_vct[col] = df[col].apply(lambda x: splitter(x, bondaries[col]))
        tmp_dict = woe_dict[col].set_index('group')['WoE'].to_dict()
        df_woe_vct[col] = df_woe_vct[col].map(tmp_dict)

    # Модельный скор, участвующий в интегральной модели. Вычисляется как линейная регрессия
    spark_score = model_sp['intercept']
    for i in range(0, len(features_spark)):
        spark_score = spark_score + model_sp['coefs'][i] * df_woe_vct.iloc[0, i]

    # Вероятность дефолта. Вычисляется из скора линейной регрессии, становясь лог регрессией
    spark_pd = score_to_pd(spark_score)
    print(spark_pd)

    pravoru_pd = model_pr.predict_proba(df[features_pravo])[:,-1].item()
    pravoru_score = logit_pd(pravoru_pd)

    # Интегральный скор
    integr_score = (model_integr['intercept'] + 
                    model_integr['coefs'][0] * spark_score + 
                    model_integr['coefs'][1] * pravoru_score)

    # Вероятность дефолта интегральной модели
    integr_pd = score_to_pd(integr_score)
    
    return render_template(
        "index.html",
        prediction_text_1 = f"Вероятность дефолта по данным экономических показателей составляет \
        {np.round(spark_pd*100, 2)} %", 
        prediction_text_2 = f"Вероятность дефолта по данным судебных исков составляет \
        {np.round(pravoru_pd*100, 2)} %",
        prediction_text = f"Вероятность дефолта обобщающей интегральной модели составляет \
        {np.round(integr_pd*100, 2)} %"
    )


if __name__ == "__main__":
    app.run('127.0.0.1', 5000)