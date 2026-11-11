# Pre-entrega 2: Pipeline de procesamiento validado

Pipeline asíncrono de extracción de entidades técnicas con Pydantic, LangChain
y OpenAI. Reutiliza la configuración validada del cliente de la pre-entrega 1.

## Instalación

Python 3.12. En Windows (PowerShell):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
```

En Linux o macOS:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
cp .env.example .env
```

Si `.env` ya existe, editarlo sin reemplazarlo. Configurar `LLM_PROVIDER=openai`
y `OPENAI_API_KEY` con una clave válida. El modelo predeterminado es
`gpt-4.1-mini`. También se reutilizan `LLM_TEMPERATURE`, `LLM_MAX_TOKENS`,
`LLM_TIMEOUT` y `LLM_MAX_RETRIES` del módulo 1.

## Ejecución

```powershell
.\.venv\Scripts\python.exe test_pipeline.py
```

En Linux o macOS: `.venv/bin/python test_pipeline.py`.
El script realiza una llamada real a OpenAI y requiere cuota disponible.

Ejemplo de salida:

```json
{
  "tecnologias": ["FastAPI", "Redis", "PostgreSQL"],
  "nivel_de_criticidad": "alta",
  "resumen_tecnico": "API con FastAPI, caché en Redis y persistencia en PostgreSQL; servicio caído por agotamiento del pool de conexiones."
}
```

`process_text(text: str)` en `chain.py` devuelve una instancia de
`EntidadesTecnicas`. La cadena combina `ChatPromptTemplate` con
`model.with_structured_output(EntidadesTecnicas)` mediante LCEL y se ejecuta
con `.ainvoke()`.

Los tres campos son obligatorios. La criticidad solo acepta `baja`, `media` o
`alta`; el resumen y los nombres de tecnologías no pueden estar vacíos.
Si el texto no menciona tecnologías, se acepta `tecnologias: []`.
Ante información ambigua, se usa criticidad `media` sin inventar tecnologías.

`.with_retry(stop_after_attempt=2)` permite un reintento ante JSON inválido,
campos faltantes o respuestas truncadas. El SDK detecta `finish_reason=length`
antes de transformar el contenido y lanza `LengthFinishReasonError`.
Si ambos intentos fallan, la excepción se propaga. Los logs muestran cada
intento y el resultado de la validación. `LLM_MAX_RETRIES` controla por separado
los reintentos de red del SDK.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

En Linux o macOS: `.venv/bin/python -m unittest discover -s tests -v`.
Las pruebas simulan HTTP sin credenciales reales. Incluyen recuperación ante
JSON mal formado, campos incompletos, truncamiento y texto ambiguo.
