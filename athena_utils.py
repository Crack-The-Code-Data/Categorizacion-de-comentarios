import boto3
import pandas as pd
import time
import io


def run_athena_query(query, name='', region='us-east-1', bucket='aws-athena-query-results-us-east-1-158862062418'):
    """
    Ejecuta una consulta en Athena, guarda el resultado en Parquet en S3, lo carga en un DataFrame y limpia los recursos.
    Args:
        query (str): Consulta SQL a ejecutar.
        region (str): Región de AWS.
        bucket (str): Bucket de salida de Athena.
    Returns:
        pd.DataFrame: DataFrame con los resultados de la consulta.
    """
    athena = boto3.client('athena', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    timestamp = int(time.time())
    parquet_output_path = f's3://{bucket}/python_ale/{name}_{timestamp}/'
    
    ctas_query = f"""
    CREATE TABLE python_table_{name}_{timestamp}
    WITH (
      format = 'PARQUET',
      external_location = '{parquet_output_path}',
      write_compression = 'SNAPPY'
    ) AS
    {query}
    """
    
    response = athena.start_query_execution(
        QueryString=ctas_query,
        QueryExecutionContext={'Database': 'datalake'},
        ResultConfiguration={'OutputLocation': f's3://{bucket}/'}
    )
    query_execution_id = response['QueryExecutionId']
    while True:
        result = athena.get_query_execution(QueryExecutionId=query_execution_id)
        state = result['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(2)
    if state == 'FAILED':
        raise Exception(result['QueryExecution']['Status'].get('StateChangeReason'))
    if state == 'SUCCEEDED':
        df = pd.read_parquet(parquet_output_path, engine='pyarrow')
        # Eliminar los archivos Parquet de S3 después de cargarlos en el DataFrame
        try:
            prefix = f'python_ale/{timestamp}/'
            objects = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
            if 'Contents' in objects:
                delete_keys = [{'Key': obj['Key']} for obj in objects['Contents']]
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={'Objects': delete_keys}
                )
        except Exception as e:
            print(f"⚠️ Error al eliminar archivos de S3: {str(e)}")
        # Eliminar la tabla en Athena
        try:
            drop_query = f"DROP TABLE IF EXISTS datalake.python_table_sentiments_{timestamp};"
            drop_resp = athena.start_query_execution(
                QueryString=drop_query,
                QueryExecutionContext={'Database': 'datalake'},
                ResultConfiguration={'OutputLocation': f's3://{bucket}/'}
            )
            drop_qid = drop_resp['QueryExecutionId']
            while True:
                drop_status = athena.get_query_execution(QueryExecutionId=drop_qid)['QueryExecution']['Status']['State']
                if drop_status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                time.sleep(1)
        except Exception as e:
            print(f"⚠️ Error al eliminar la tabla en Athena: {e}")
        return df
    else:
        raise Exception(result['QueryExecution']['Status'].get('StateChangeReason')) 
    



def run_athena_query(query: str, name: str = '', region: str = 'us-east-1', bucket: str = 'data-lake-athena-querys') -> pd.DataFrame:
    """
    Ejecuta una consulta en Athena, guarda el resultado en Parquet en S3, lo carga en un DataFrame y limpia los recursos.

    Si la consulta no devuelve filas, retorna un DataFrame vacío.
    En caso de error, asegura la eliminación de recursos (S3 y tabla) antes de propagar la excepción.
    """
    athena = boto3.client('athena', region_name=region)
    s3 = boto3.client('s3', region_name=region)
    timestamp = int(time.time())
    table_name = f"python_table_{name}_{timestamp}"
    s3_prefix = f"python/temporales/{name}_{timestamp}/"
    parquet_output_path = f's3://{bucket}/{s3_prefix}'

    ctas_query = f"""
    CREATE TABLE {table_name}
    WITH (
      format = 'PARQUET',
      external_location = '{parquet_output_path}',
      write_compression = 'SNAPPY'
    ) AS
    {query}
    """

    # Helper to clean up resources
    def clean_up():
        # Eliminar archivos Parquet de S3
        try:
            objs = s3.list_objects_v2(Bucket=bucket, Prefix=s3_prefix)
            if objs.get('Contents'):
                delete_keys = [{'Key': obj['Key']} for obj in objs['Contents']]
                s3.delete_objects(Bucket=bucket, Delete={'Objects': delete_keys})
        except ClientError as e:
            print(f"⚠️ Error al eliminar archivos de S3: {e}")

        # Eliminar la tabla en Athena
        try:
            drop_query = f"DROP TABLE IF EXISTS {table_name};"
            resp = athena.start_query_execution(
                QueryString=drop_query,
                QueryExecutionContext={'Database': 'datalake'},
                ResultConfiguration={'OutputLocation': f's3://{bucket}/'}
            )
            drop_qid = resp['QueryExecutionId']
            # Esperar a que termine el DROP
            while True:
                st = athena.get_query_execution(QueryExecutionId=drop_qid)['QueryExecution']['Status']['State']
                if st in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                    break
                time.sleep(1)
        except ClientError as e:
            print(f"⚠️ Error al eliminar la tabla en Athena: {e}")

    # Ejecutar CTAS y procesar resultados
    try:
        resp = athena.start_query_execution(
            QueryString=ctas_query,
            QueryExecutionContext={'Database': 'datalake'},
            ResultConfiguration={'OutputLocation': f's3://{bucket}/'}
        )
        qid = resp['QueryExecutionId']
        # Esperar a que termine la ejecución
        while True:
            status = athena.get_query_execution(QueryExecutionId=qid)['QueryExecution']['Status']['State']
            if status in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
                break
            time.sleep(2)

        if status != 'SUCCEEDED':
            raise Exception(athena.get_query_execution(QueryExecutionId=qid)['QueryExecution']['Status'].get('StateChangeReason'))

        # Intentar leer los Parquet; si no existen, devolver df vacío
        try:
            df = pd.read_parquet(parquet_output_path, engine='pyarrow')
        except (FileNotFoundError, OSError, ValueError):
            df = pd.DataFrame()

        return df

    except Exception as e:
        # En caso de error, limpiar recursos y propagar
        clean_up()
        raise

    finally:
        # Siempre limpiar al final si hubo éxito o excepción
        clean_up()



def export_dataframe_to_s3_json(df, name, bucket='raw-data-lake-virginia', key='python/category_analysis', region='us-east-1', orient='records'):
    """
    Exporta un DataFrame como JSON y lo sube a un bucket de S3.

    Args:
        df (pd.DataFrame): DataFrame a exportar.
        name (str): Nombre del archivo JSON (sin .json).
        bucket (str): Nombre del bucket de S3.
        key (str): Carpeta dentro del bucket donde guardar el archivo.
        region (str): Región de AWS.
        orient (str): Formato de exportación JSON (por defecto 'records').
    """
    s3 = boto3.client('s3', region_name=region)
    try:
        buffer = io.StringIO()
        df.to_json(buffer, orient=orient, lines=True, force_ascii=False, date_format='iso')

        s3_key = f"{key}/{name}.json"  # <-- construir el path completo

        s3.put_object(Bucket=bucket, Key=s3_key, Body=buffer.getvalue().encode('utf-8'))
        print(f"✅ JSON exportado correctamente a s3://{bucket}/{s3_key}")
    except Exception as e:
        print(f"❌ Error al exportar JSON a S3: {str(e)}")



def create_athena_table(table_name, s3_location, columns, database='datalake',
                        file_format='JSON', region='us-east-1', bucket='aws-athena-query-results-us-east-1-158862062418'):
    athena = boto3.client('athena', region_name=region)

    columns_def = ',\n  '.join([f'{name} {dtype}' for name, dtype in columns])

    file_format = file_format.upper()
    if file_format == 'JSON':
        serde = "org.openx.data.jsonserde.JsonSerDe"
        row_format = f"ROW FORMAT SERDE '{serde}'"
        stored_as = "STORED AS TEXTFILE"
    elif file_format == 'CSV':
        serde = "org.apache.hadoop.hive.serde2.OpenCSVSerde"
        row_format = f"ROW FORMAT SERDE '{serde}' WITH SERDEPROPERTIES ('separatorChar' = ',')"
        stored_as = "STORED AS TEXTFILE"
    elif file_format == 'PARQUET':
        row_format = ""
        stored_as = "STORED AS PARQUET"
    else:
        raise ValueError("Formato no soportado. Usar: JSON, CSV o PARQUET.")

    query = f"""
    CREATE EXTERNAL TABLE IF NOT EXISTS {database}.{table_name} (
      {columns_def}
    )
    {row_format}
    {stored_as}
    LOCATION '{s3_location}'
    TBLPROPERTIES ('has_encrypted_data'='false');
    """

    response = athena.start_query_execution(
        QueryString=query,
        QueryExecutionContext={'Database': database},
        ResultConfiguration={'OutputLocation': f's3://{bucket}/'}
    )

    execution_id = response['QueryExecutionId']
    while True:
        result = athena.get_query_execution(QueryExecutionId=execution_id)
        state = result['QueryExecution']['Status']['State']
        if state in ['SUCCEEDED', 'FAILED', 'CANCELLED']:
            break
        time.sleep(2)

    if state == 'SUCCEEDED':
        print(f"✅ Tabla '{table_name}' creada en Athena.")
    else:
        reason = result['QueryExecution']['Status'].get('StateChangeReason', '')
        print(f"❌ Error al crear la tabla: {state} - {reason}")

def columns_tupla(df):
    """
    Crea una tabla externa en Athena basándose en el esquema de un DataFrame.

    Args:
        df (pd.DataFrame): DataFrame con el esquema base.
        table_name (str): Nombre de la tabla a crear en Athena.
        s3_location (str): Ruta S3 donde están los archivos (ej: 's3://mi-bucket/mis-datos/').
        database (str): Base de datos en Athena.
        file_format (str): Formato de archivo: 'JSON', 'PARQUET' o 'CSV'.
        region (str): Región AWS.
        bucket (str): Bucket donde se guarda el log de ejecución.
    """
    import numpy as np

    # Mapear tipos de pandas a tipos de Athena
    type_map = {
        'int64': 'bigint',
        'int32': 'int',
        'float64': 'double',
        'bool': 'boolean',
        'datetime64[ns]': 'timestamp',
        'object': 'string',
        'string': 'string',
        'category': 'string',
        'float32': 'float',
        'timedelta[ns]': 'string',
    }

    columns = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        athena_type = type_map.get(dtype, 'string')  # default a string si no se reconoce
        columns.append((col, athena_type))
    
    return columns
