# ! pip install spanish_sentiment_analysis

from classifier import * 

import pandas as pd
import streamlit as st
import time

nlp = SentimentClassifier()

def main():

    st.title('Coke.ai')
    st.title('Análisis de sentimiento spanish-entiment-analysis')

    # text = st.text_input("Expresión:")
    write_here = "Texto aqui..."
    text = st.text_area("Incluya un texto ..", write_here)
    if st.button("Analizar"):
        if text != write_here:
            label, score = sentimiento(text)
            st.success('Sentimiento de ['+ text + ']')
            st.success(label)
            st.success('%.2f' % score)    
        else:
            st.error("Ingresa un texto y presiona el boton Analizar ..")
    else:
        st.info(
            "Ingresa un texto y presiona el boton Analizar .."
        )

    uploaded_file = st.file_uploader("O bien puede seleccionar un archivo CSV para procesar hasta 3500 párrafos (se procesará columna 'text')",type=['csv'])
    if uploaded_file is not None:
        if st.button("Procesar Archivo CSV"):
            data = pd.read_csv(uploaded_file,usecols=["text"])
            #pd.read_parquet("penguin-dataset.parquet")
            #data.to_parquet("penguin-dataset.parquet")
            st.success("Procesando CSV ..")
            data[data['text'].str.strip().astype(bool)]
            data['text'] = data['text'].astype(str)
            total_reg = len(data)
            t0 = time.time()
            msg = f"Espere por favor, esto puede tomar algun tiempo .. procesando {total_reg:.0f} elementos"  if total_reg>1000 else f"Espere .. procesando {total_reg:.0f} registros"
            with st.spinner(msg):
                g = lambda x: pd.Series(sentimiento(x.text))
                data[['label', 'score']] = data.apply(g, axis=1)    
            csv = convert_df(data)
            st.success(f'{total_reg:.0f} registros procesados con éxito en {time.time() - t0:.0f} seg')
            if st.download_button(label="Presione para descargar archivo procesado", data=csv, file_name='_cokeai_results.csv', mime='text/csv'):
                st.success("Descargado con éxito ..")
                st.stop()
        else:
            st.error("Aun no se ha procesado el archivo..")
    else:
        st.info("Aun no se ha procesado el archivo ..")

@st.cache
def sentimiento(text):
    #try:
        conditions = {
            1: 'Muy Malo',
            2: 'Malo',
            3: 'Neutro',
            4: 'Bueno',
            5: 'Muy bueno'
        }
        result = nlp.predict(text)
        label = conditions[CheckForLess([0.1, 0.2, 0.5, 0.8, 1],result)]
        return label, round(result,4)
    #except:
    #    return "_Error", -1

@st.cache
def CheckForLess(list1, val): 
      
    # traverse in the list
    i=1
    for x in list1: 
          if val <= x: 
            return i
          else:
            i=i+1
    return False

@st.cache
def convert_df(df):
    # Cache the conversion to prevent computation on every rerun
    return df.to_csv().encode('utf-8')


if __name__ == '__main__':
    main()