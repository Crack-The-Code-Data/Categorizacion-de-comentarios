import pandas as pd
import openai
import os
import json
import tiktoken
import time
from datetime import datetime
from dotenv import load_dotenv
from tqdm import tqdm
from typing import Dict, Any
import re
import concurrent.futures
import threading


# Cargar variables de entorno desde .env
load_dotenv()

# Configurar la clave API de OpenAI
env_api_key = os.getenv('API_KEY')
if not env_api_key:
    raise ValueError("No se encontró la variable de entorno API_KEY. Asegúrate de que tu archivo .env está configurado correctamente.")
openai.api_key = env_api_key

# Métricas Acumulativas y lock para concurrencia
total_input_tokens = 0
total_output_tokens = 0
registro_tokens = []
token_lock = threading.Lock()

# Constants for token limits
DEFAULT_MAX_TOKENS = 9000
# Calculate maximum tokens for the prompt: total limit minus what we allocate for completion
MAX_INPUT_TOKENS = 4000

# Diccionario de precios por modelo (USD por 1M tokens)
PRECIOS_MODELOS = {
    'gpt-4o-mini': {'input': 0.15, 'output': 0.60},
    'gpt-4o': {'input': 5.00, 'output': 15.00},
}

PROMPT_CATEGORIZACION='''
**Tarea de Análisis y Categorización de Feedback**
**1. Objetivo:**
Tu tarea es analizar un lote de respuestas de estudiantes y asignar a cada una la categoría más adecuada de la guía de codificación proporcionada. Debes basar tu análisis en el contexto de la pregunta (`question_name`) y el contenido de la respuesta (`answer`).
**2. Formato de Entrada:**
Recibirás los datos como un array de objetos JSON. Cada objeto contiene un `id` único, `question_name` y `answer`.

**3. Guía de Codificación Unificada:**
Etiquetado Múltiple: Una misma respuesta puede (y a menudo debe) ser asignada a varias categorías si toca diferentes temas. Ejemplo: "El profesor explica bien pero la plataforma es confusa" se codifica tanto en la categoría de docente como en la de plataforma.


---

**Eje: Contenido y Materiales**

1.  **Contenido claro y fácil de entender**
    *   **Enfoque:** La claridad y estructura de los materiales de aprendizaje (actividades, lecturas, videos, etc.), no la explicación del docente.
    *   **Inclusiones:** "Entendí todo", "las instrucciones eran claras", "el material es fácil de seguir", "el programa está bien organizado".
    *   **Exclusiones:** Comentarios específicos sobre la habilidad del docente para explicar (eso corresponde a la categoría 7).

2.  **Contenido útil y aplicable a mi carrera**
    *   **Enfoque:** El valor práctico y la relevancia del conocimiento para el futuro profesional, académico o personal del estudiante.
    *   **Inclusiones:** "Me servirá para mi trabajo", "es útil para mi carrera", "lo puedo aplicar en mi emprendimiento", "aprendo cosas para la vida".
    *   **Exclusiones:** Comentarios generales sobre si le "gusta" el contenido sin mencionar su utilidad (eso es categoría 3).

3.  **Contenido entretenido y motivador**
    *   **Enfoque:** El disfrute, el engagement y el interés que genera el contenido o las actividades.
    *   **Inclusiones:** "Me divertí", "la clase fue chévere", "las actividades eran interesantes", "me gusta el tema".
    *   **Exclusiones:** Si el elogio es específico a la dinámica creada por el docente (ej: "el profe hace la clase divertida"), priorizar la categoría del docente (7 u 8).

4.  **Contenido confuso o difícil de seguir**
    *   **Enfoque:** Dificultad intelectual para comprender los temas, conceptos o instrucciones de las actividades. No se refiere a problemas técnicos.
    *   **Inclusiones:** "No entendí el tema", "me pareció muy complicado", "me perdí", "la información era densa".
    *   **Exclusiones:** Problemas con la plataforma (usar 15), quejas sobre el ritmo del docente (usar 10), o falta de interés (usar 5 o 6).

5.  **Contenido aburrido o monótono**
    *   **Enfoque:** Quejas de aburrimiento o falta de dinamismo relacionadas con el contenido o la estructura de la clase en general.
    *   **Inclusiones:** "La clase es aburrida", "siempre hacemos lo mismo", "me da sueño", "es muy monótono".
    *   **Exclusiones:** Si la crítica apunta directamente a la metodología del docente (ej: "el profesor es aburrido"), usar la categoría 10.

6.  **Contenido sin relevancia para mis objetivos**
    *   **Enfoque:** El estudiante siente que el contenido no se alinea con sus metas o intereses profesionales o personales.
    *   **Inclusiones:** "Esto no me sirve para mi carrera", "no lo usaré en el futuro", "esperaba aprender otra cosa".
    *   **Exclusiones:** Simples quejas de "no me gusta" (eso es categoría 20).

---

**Eje: Docente**

7.  **Buen nivel de explicación del docente**
    *   **Enfoque:** La habilidad del docente para explicar, comunicar ideas de forma clara y facilitar la comprensión.
    *   **Inclusiones:** "Explica muy bien", "es muy claro", "se le entiende todo", "hace que lo difícil parezca fácil".
    *   **Exclusiones:** Elogios a su paciencia o amabilidad (usar 9) o a su conocimiento general (usar 8).

8.  **Docente experto y con dominio del tema**
    *   **Enfoque:** La percepción del estudiante sobre el profundo conocimiento y la experiencia del docente en la materia.
    *   **Inclusiones:** "Sabe mucho", "domina el tema", "se nota que tiene experiencia", "es un experto".
    *   **Exclusiones:** Comentarios sobre su habilidad para explicar (usar 7). Un docente puede saber mucho pero no explicar bien.

9.  **Docente amable y paciente al resolver dudas**
    *   **Enfoque:** La actitud y disposición del docente para ayudar, su trato y su paciencia.
    *   **Inclusiones:** "Resuelve todas mis dudas", "es muy paciente", "nos ayuda mucho", "es muy amable y cercano".
    *   **Exclusiones:** Elogios a su capacidad de explicación general (usar 7).

10. **Docente con método poco dinámico o poco claro**
    *   **Enfoque:** Crítica a la metodología, ritmo o claridad de la enseñanza del docente.
    *   **Inclusiones:** "Explica muy rápido", "no se le entiende", "su clase es monótona", "se desvía del tema", "no es dinámico".
    *   **Exclusiones:** Quejas sobre el contenido en sí (usar 4 o 6) o sobre su disposición a ayudar (usar 12).

11. **Docente que demuestra falta de conocimiento**
    *   **Enfoque:** El estudiante percibe que el profesor tiene lagunas en su conocimiento o no domina un tema.
    *   **Inclusiones:** "El profe no sabía la respuesta", "parecía inseguro", "tuvo que buscarlo en Google".
    *   **Exclusiones:** Dificultad para explicar un tema que sí domina (usar 10).

12. **Docente poco dispuesto a ayudar**
    *   **Enfoque:** Crítica a la actitud del docente, su falta de paciencia o su poca disposición para resolver dudas.
    *   **Inclusiones:** "No ayuda", "no responde las preguntas", "se enoja si le preguntamos", "no tiene paciencia".

---

**Eje: Técnico y Plataforma**

13. **Problemas técnicos**
    *   **Enfoque:** Fallos objetivos de hardware o conectividad.
    *   **Inclusiones:** "Mala conexión a internet", "no se escuchaba el micrófono", "la cámara no funcionaba", "se trababa la llamada".
    *   **Exclusiones:** Problemas de usabilidad de la plataforma (usar 15).

14. **Plataforma intuitiva y rica en recursos**
    *   **Enfoque:** La experiencia de usuario con la plataforma (Campus CTC, etc.) es positiva, fácil y fluida.
    *   **Inclusiones:** "La plataforma es fácil de usar", "es muy intuitiva", "encuentro todo rápido", "es sencilla y clara".
    *   **Exclusiones:** Elogios al contenido que está *dentro* de la plataforma (usar 1, 2 o 3).

15. **Plataforma confusa o con fallos técnicos**
    *   **Enfoque:** La plataforma es difícil de navegar, poco intuitiva o presenta errores de funcionamiento.
    *   **Inclusiones:** "No entiendo la plataforma", "es complicada", "me pierdo", "no funciona el botón", "se cuelga".
    *   **Exclusiones:** Dificultad para entender el contenido académico (usar 4).

---

**Eje: Percepción General y Sugerencias**

16. **Proyecto motivador**
    *   **Enfoque:** El estudiante considera que el programa en su conjunto fue una experiencia que estimuló su creatividad e interés.
    *   **Inclusiones:** "Fue un reto que me gustó", "me motivó a crear", "fue un proyecto muy interesante".
    *   **Exclusiones:** Comentarios positivos sobre partes específicas (usar categorías de contenido o docente).

17. **Proyecto desmotivador**
    *   **Enfoque:** El estudiante expresa frustración o confusión general con los objetivos o la ejecución del programa.
    *   **Inclusiones:** "No entendí el propósito del proyecto", "fue muy complicado de realizar", "me sentí perdido".
    *   **Exclusiones:** Críticas a componentes específicos (usar categorías de contenido, docente o plataforma).

18. **Sugerencias y propuestas de mejora**
    *   **Enfoque:** Comentarios que proponen cambios, críticas constructivas o ideas para mejorar cualquier aspecto del programa.
    *   **Inclusiones:** "Deberían añadir más ejemplos", "sugiero que las clases duren más", "sería mejor si fuera presencial", "le falta profundizar en X tema".
    *   **Exclusiones:** Quejas puras sin una propuesta implícita (ej: "no me gustó" va en 20).

19. **Comentarios positivos generales**
    *   **Enfoque:** Expresiones de satisfacción general sin dar detalles específicos. Se usa cuando no hay suficiente información para clasificar en una categoría más precisa.
    *   **Inclusiones:** "Me gusta mucho", "excelente", "estoy satisfecho", "todo bien", "perfecto".
    *   **Exclusiones:** Cualquier comentario que dé un mínimo detalle sobre *qué* es bueno (ej: "explican bien" o "buen profe" -> 7).

20. **Comentarios negativos generales**
    *   **Enfoque:** Expresiones de insatisfacción general sin detalles específicos.
    *   **Inclusiones:** "No me gusta", "no me sirve", "es malo", "no estoy conforme".
    *   **Exclusiones:** Cualquier comentario que dé un mínimo detalle sobre *qué* es malo (ej: "es aburrido" -> 5).

21. **Otro**
    *   **Enfoque:** Usar **únicamente** como último recurso para respuestas que son imposibles de clasificar.
    *   **Inclusiones:** "porque si", "no tengo", "normal", "ninguna", "mi experiencia", "N/A", respuestas en blanco, o comentarios completamente fuera de contexto (ej: "me gusta el fútbol").
    *   **Exclusiones:** Cualquier respuesta que tenga un mínimo de intención o sentimiento interpretable.


**4. Formato de Salida Obligatorio:**
Tu respuesta DEBE ser únicamente un array de objetos JSON válido, sin texto adicional antes o después. Cada objeto debe contener el `id` original y la `category` que asignaste.
**Ejemplo de Salida:**
```json
[
  {
    "id": 1,
    "category": ["Buen nivel de explicación del docente"]
  },
  {
    "id": 2,
    "category": ["Docente poco dispuesto a ayudar", "Contenido claro y fácil de entender"]
  }
]
```


'''

# Helper: crear codificador y contar tokens
ENCODING = tiktoken.encoding_for_model("gpt-4o-mini")

def count_tokens(text: str) -> int:
    return len(ENCODING.encode(text))

# GPT API call
def call_gpt(prompt: str, modelo: str = "gpt-4o-mini", max_tokens: int = DEFAULT_MAX_TOKENS, temperature: float = 0.5) -> dict:
    try:
        response = openai.chat.completions.create(
            model=modelo,
            messages=[
                {"role": "system", "content": PROMPT_CATEGORIZACION},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response
    except openai.OpenAIError as e:
        print(f"Error en la API de OpenAI: {e}")
        return None
    except Exception as e:
        print(f"Error inesperado: {e}")
        return None

# Agrupar registros por tamaño en tokens de manera eficiente
def split_batches_fast(df: pd.DataFrame) -> list:
    batches = []
    current_batch = []
    current_token_count = count_tokens(PROMPT_CATEGORIZACION) + count_tokens("Por favor, categoriza el siguiente lote de respuestas: []")

    for idx, row in df.iterrows():
        record = {"id": int(idx), "question_name": row['question_name'], "answer": row['answer']}
        record_json = json.dumps(record, ensure_ascii=False)
        record_tokens = count_tokens(record_json) + 2  # 2 extra por coma y corchetes

        if current_token_count + record_tokens > MAX_INPUT_TOKENS:
            batches.append(current_batch)
            current_batch = [record]
            current_token_count = count_tokens(PROMPT_CATEGORIZACION) + count_tokens("Por favor, categoriza el siguiente lote de respuestas: []") + record_tokens
        else:
            current_batch.append(record)
            current_token_count += record_tokens

    if current_batch:
        batches.append(current_batch)

    return batches

def extract_json_string(llm_output: str) -> str:
    # 1. Quitar etiquetas de bloque de código
    for fence in ["```json", "```", "``` js", "``` txt"]:
        if fence in llm_output:
            llm_output = llm_output.replace(fence, "")
    # 2. Strips finales
    return llm_output.strip()


JSON_ARRAY_RE = re.compile(r"(\[.*\])", re.DOTALL)

def find_json_array(text: str) -> str:
    match = JSON_ARRAY_RE.search(text)
    if not match:
        raise ValueError("No se encontró un array JSON en la respuesta")
    return match.group(1)

def get_json_chunk(llm_output: str) -> str:
    cleaned = extract_json_string(llm_output)
    return find_json_array(cleaned)


# Categorizar el DataFrame en lotes optimizados por tokens
def categorizar_dataframe(df: pd.DataFrame,
                          model: str = "gpt-4o-mini",
                          max_token: int = DEFAULT_MAX_TOKENS,
                          parallel_calls: int = 1,
                          verbose: bool = False,
                          progress: bool = True) -> pd.DataFrame:
    """
    Categoriza un DataFrame de respuestas en paralelo usando OpenAI.

    Args:
        df: DataFrame con columnas 'question_name' y 'answer'.
        model: Modelo OpenAI a usar.
        max_token: Máximo tokens de respuesta.
        parallel_calls: Número máximo de llamadas concurrentes (1 = secuencial, hasta 10).
    """
    global total_input_tokens, total_output_tokens, registro_tokens

    if 'question_name' not in df.columns or 'answer' not in df.columns:
        raise ValueError("El DataFrame debe contener 'question_name' y 'answer'.")

    registro_tokens.clear()
    batches = split_batches_fast(df)
    all_results = []

    # Worker para procesar un lote
    def process_batch(lote):
        nonlocal all_results
        try:
            if verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ⏳ Empezando lote en hilo {threading.current_thread().name}")
            input_json = json.dumps(lote, ensure_ascii=False)
            prompt_lote = f"Por favor, categoriza el siguiente lote de respuestas: {input_json}"
            response = call_gpt(prompt=prompt_lote, modelo=model, max_tokens=max_token)
            if not response:
                return pd.DataFrame()

            # Actualizar métricas bajo lock
            usage = response.usage
            with token_lock:
                registro_tokens.append({
                    'fecha_hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'modelo': response.model,
                    'input_tokens': usage.prompt_tokens,
                    'output_tokens': usage.completion_tokens,
                })

            llm_output = response.choices[0].message.content.strip()
            llm_output = get_json_chunk(llm_output)
            parsed = json.loads(llm_output)
            if verbose:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] ✅ Terminó lote en hilo {threading.current_thread().name}")
            return pd.DataFrame(parsed)

        except Exception as e:
            try:
                tqdm.write(f"Error en lote: {e}")
            except Exception:
                print(f"Error en lote: {e}")
            return pd.DataFrame()

    # Ejecutar secuencial o paralelo según parámetro
    if parallel_calls > 1:
        max_workers = min(parallel_calls, 10)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_batch, lote) for lote in batches]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Categorizando lotes en paralelo", leave=False, dynamic_ncols=True, disable=not progress):
                df_batch = future.result()
                if not df_batch.empty:
                    all_results.append(df_batch)
    else:
        for lote in tqdm(batches, desc="Categorizando lotes secuencialmente", leave=False, dynamic_ncols=True, disable=not progress):
            df_batch = process_batch(lote)
            if not df_batch.empty:
                all_results.append(df_batch)

    # Concatenar resultados y mapear categorías
    if all_results:
        combined = pd.concat(all_results)
        if verbose:
            print("¿Hay IDs duplicados en los resultados?", combined['id'].duplicated().any())
        result_df = pd.concat(all_results).set_index('id')
    else:
        result_df = pd.DataFrame(columns=['category'])

    df_out = df.copy()
    result_df = result_df[~result_df.index.duplicated(keep='first')]

    df_out['categoria'] = df.index.map(lambda i: result_df.at[i, 'category'] if i in result_df.index else [])
    return df_out


def guardar_metricas(filepath: str = 'metricas_uso_openai.csv'):
    """
    Guarda solo la fila con el total de la ejecución actual.
    """
    if not registro_tokens:
        print("No hay métricas nuevas para guardar.")
        return

    metricas_df = pd.DataFrame(registro_tokens)
    # Tomar el modelo de la última ejecución registrada
    modelo = 'gpt-4o-mini'

    # Buscar el modelo exacto en el diccionario de precios
    precios = PRECIOS_MODELOS.get(modelo)
    # Si no está, buscar la versión "padre" (por ejemplo, 'gpt-4o' para 'gpt-4o-mini')
    if precios is None:
        modelo_base = '-'.join(modelo.split('-')[:2])
        precios = PRECIOS_MODELOS.get(modelo_base, {'input': 0, 'output': 0})

    # Crear solo la fila del total de la ejecución actual
    total_ejecucion = {
        'fecha_hora': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'modelo': modelo,
        'input_tokens': metricas_df['input_tokens'].sum(),
        'output_tokens': metricas_df['output_tokens'].sum(),
    }

    # Guardar solo la fila del total
    total_ejecucion_df = pd.DataFrame([total_ejecucion])
    total_ejecucion_df['costo_usd'] = (
        (total_ejecucion_df['input_tokens'] * precios['input']) +
        (total_ejecucion_df['output_tokens'] * precios['output'])
    ) / 1_000_000

    if os.path.exists(filepath):
        total_ejecucion_df.to_csv(filepath, mode='a', header=False, index=False)
    else:
        total_ejecucion_df.to_csv(filepath, mode='w', header=True, index=False)

    print(f"Solo el total de la ejecución guardado en '{filepath}'")
    print(f"Modelo: {total_ejecucion['modelo']}")
    print(f"Tokens enviados: {total_ejecucion['input_tokens']}")
    print(f"Tokens recibidos: {total_ejecucion['output_tokens']}")