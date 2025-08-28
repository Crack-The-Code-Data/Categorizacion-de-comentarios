
import pandas as pd
import json
import numpy as np
import athena_utils as au
import openia_script as OA
import os

def main():
    """
    Script principal para categorizar respuestas de encuestas de satisfacción.
    1. Carga datos de respuestas y categorías existentes desde Athena.
    2. Identifica respuestas nuevas no categorizadas.
    3. Usa OpenAI para asignar categorías a las nuevas respuestas.
    4. Procesa, limpia y mapea las categorías obtenidas.
    5. Guarda las métricas de uso de la API de OpenAI.
    6. Exporta el DataFrame final con todas las categorías a S3.
    """
    print("Iniciando el proceso de categorización de respuestas.")

    # 2. Definición de Consultas SQL
    query_sentiment_analysis = """
    SELECT
        ceq.tag,
        sa.* 
    FROM
       sentiment_analysis sa
       left join moodle_course_evaluation_questions ceq on (sa.question_id = ceq.question_id and sa.unique_id = ceq.unique_id)
    where 
       lower(sa.question_name) not like '%indica%'
       and lower(sa.question_name) not like '%qué tan fácil%';
    """

    query_category_response = """
    SELECT
       *
    FROM
       response_satisfaccion_category;
    """

    # 3. Carga de Datos desde Athena
    print("Cargando datos desde Athena...")
    df_sentiment = au.run_athena_query(query_sentiment_analysis, 'sentiment')
    df_sentiment = df_sentiment[df_sentiment['answer'].str.split().str.len() > 1]
    
    try:
        df_categorias = au.run_athena_query(query_category_response, 'category_response')
    except Exception as e:
        print(f"No se pudieron cargar categorías existentes o la tabla no existe. Se procederá sin ellas. Error: {e}")
        df_categorias = pd.DataFrame()

    print(f"Se encontraron {len(df_sentiment)} respuestas para analizar.")
    if not df_categorias.empty:
        print(f"Se cargaron {len(df_categorias)} categorías existentes.")

    # Comparar para encontrar respuestas sin procesar
    columnas_comparar = ['moodle_id', 'unique_id', 'activity_id', 'attempt_id', 'question_id', 'answer']
    
    if df_categorias.empty:
        df_inprocess = df_sentiment.copy()
    else:
        df_categorias = df_categorias[df_categorias['answer'].notnull() & (df_categorias['answer'] != '')]
        df_inprocess = df_sentiment.merge(
            df_categorias[columnas_comparar],
            on=columnas_comparar,
            how='left',
            indicator=True
        )
        df_inprocess = df_inprocess[df_inprocess['_merge'] == 'left_only']
        df_inprocess = df_inprocess.drop(columns=['_merge'])

    if df_inprocess.empty:
        print("No hay respuestas nuevas para procesar. Finalizando el script.")
        return

    print(f"Se procesarán {len(df_inprocess)} respuestas nuevas.")

    # 4. Procesamiento y Categorización con OpenAI
    print("Iniciando la categorización con OpenAI...")
    df_inprocess = OA.categorizar_dataframe(df=df_inprocess, parallel_calls=8)
    
    print(f"Respuestas sin categoría asignada por OpenAI: {len(df_inprocess[df_inprocess['categoria'] == 'Sin categoría'])}")

    # Mapeo y limpieza de categorías
    mapeo_categorias = {
        "Contenido claro y fácil de entender": ["Contenido", "Positivo"],
        "Contenido útil y aplicable a mi carrera": ["Contenido", "Positivo"],
        "Contenido entretenido y motivador": ["Contenido", "Positivo"],
        "Contenido confuso o difícil de seguir": ["Contenido", "Negativo"],
        "Contenido aburrido o monótono": ["Contenido", "Negativo"],
        "Contenido sin relevancia para mis objetivos": ["Contenido", "Negativo"],
        "Buen nivel de explicación del docente": ["Docente", "Positivo"],
        "Docente experto y con dominio del tema": ["Docente", "Positivo"],
        "Docente amable y paciente al resolver dudas": ["Docente", "Positivo"],
        "Docente con método poco dinámico o poco claro": ["Docente", "Negativo"],
        "Docente que demuestra falta de conocimiento": ["Docente", "Negativo"],
        "Docente poco dispuesto a ayudar": ["Docente", "Negativo"],
        "Problemas técnicos": ["Programa", "Negativo"],
        "Plataforma intuitiva y rica en recursos": ["Plataforma", "Positivo"],
        "Plataforma confusa o con fallos técnicos": ["Plataforma", "Negativo"],
        "Proyecto motivador": ["Programa", "Positivo"],
        "Proyecto desmotivador": ["Programa", "Negativo"],
        "Sugerencias y propuestas de mejora": ["Programa", "Positivo"],
        "Comentarios positivos generales": ["Programa", "Positivo"],
        "Comentarios negativos generales": ["Programa", "Negativo"],
        "Otro": ["Otro", "Neutro"],
        "Comentarios positivos generales (docente)": ["Docente", "Positivo"],
        "Comentarios positivos generales (programa)": ["Programa", "Positivo"],
        "Comentarios positivos generales (contenido)": ["Contenido", "Positivo"],
        "Comentarios positivos generales (campus)": ["Plataforma", "Positivo"],
        "Comentarios negativos generales (docente)": ["Docente", "Negativo"],
        "Comentarios negativos generales (programa)": ["Programa", "Negativo"],
        "Comentarios negativos generales (contenido)": ["Contenido", "Negativo"],
        "Comentarios negativos generales (campus)": ["Plataforma", "Negativo"]
    }

    def agregar_tag_a_categoria(row):
        categoria = row['categoria']
        tag_raw = row.get('tag', None)
        question_raw = row.get('question_name', '')
        tag = str(tag_raw).strip().lower() if tag_raw is not None else ''
        question = str(question_raw).strip().lower()
        texto = tag if tag not in ('', 'nan', 'none') else question
        if categoria in ['Comentarios positivos generales', 'Comentarios negativos generales']:
            if 'contenido' in texto:
                categoria += ' (contenido)'
            elif 'docente' in texto:
                categoria += ' (docente)'
            elif 'programa' in texto:
                categoria += ' (programa)'
            elif 'campus' in texto:
                categoria += ' (campus)'
        return categoria

    df_inprocess['categoria'] = df_inprocess['categoria'].apply(lambda x: x if isinstance(x, list) else [x])
    df_explode = df_inprocess.explode('categoria').explode('categoria')
    
    df_explode['categoria_tipo'] = df_explode['categoria'].map(lambda x: mapeo_categorias.get(x, [None, None])[0])
    df_explode['categoria_sentimiento'] = df_explode['categoria'].map(lambda x: mapeo_categorias.get(x, [None, None])[1])
    df_explode['categoria'] = df_explode['categoria'].str.split('_').str[0]
    
    df_explode = df_explode[df_explode['categoria'] != 'Sin categoría']
    df_explode = df_explode[df_explode['categoria'].notna() & df_explode['categoria'].notnull()]
    
    df_explode['categoria'] = df_explode.apply(agregar_tag_a_categoria, axis=1)
    
    df_explode['categoria_tipo'] = df_explode['categoria'].map(lambda x: mapeo_categorias.get(x, [None, None])[0])
    df_explode['categoria_sentimiento'] = df_explode['categoria'].map(lambda x: mapeo_categorias.get(x, [None, None])[1])
    
    initial_count = len(df_explode)
    df_explode = df_explode[df_explode['categoria'].isin(mapeo_categorias.keys())]
    print(f"{initial_count - len(df_explode)} categorías eliminadas por no existir en el mapeo.")

    # Combinar con datos previamente categorizados
    if not df_categorias.empty:
        df_final = pd.concat([df_categorias, df_explode], ignore_index=True)
    else:
        df_final = df_explode

    print(f"El DataFrame final contiene {len(df_final)} filas.")

    # 5. Guardar métricas de OpenAI
    print("Guardando métricas de uso de OpenAI...")
    OA.guardar_metricas()

    # 6. Exportar a S3
    print("Exportando DataFrame final a S3...")
    au.export_dataframe_to_s3_json(df_final, 'moodle_category_responses')

    print("Proceso de categorización completado con éxito.")

if __name__ == "__main__":
    # Cargar variables de entorno
    from dotenv import load_dotenv
    load_dotenv()
    
    # Verificar que las claves de API estén presentes
    if not os.getenv('API_KEY'):
        raise ValueError("La variable de entorno API_KEY de OpenAI no está configurada.")
    if not (os.getenv('AWS_ACCESS_KEY_ID') and os.getenv('AWS_SECRET_ACCESS_KEY')):
        print("Advertencia: Las credenciales de AWS (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY) no están configuradas como variables de entorno. Se intentará usar el rol de IAM si está disponible.")

    main()
