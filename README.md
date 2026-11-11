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
El prompt indica usar criticidad `media` ante impacto incierto, sin inventar
tecnologías.

`.with_retry(stop_after_attempt=2)` permite un reintento ante JSON inválido,
campos faltantes o respuestas truncadas. El SDK detecta `finish_reason=length`
antes de transformar el contenido y lanza `LengthFinishReasonError`.
Si ambos intentos fallan, la excepción se propaga. Los logs muestran cada
intento y el resultado de la validación. `LLM_MAX_RETRIES` controla por separado
los reintentos de red del SDK.

Cada procesamiento lleva un identificador que se conserva durante el reintento.
Los logs indican el número de intento, su duración y si la validación se recuperó.
Esto permite seguir llamadas concurrentes sin registrar el texto de entrada ni
las credenciales.

## Uso y manejo de errores

`process_text()` devuelve el objeto validado o lanza una excepción:

- `ValueError`: texto vacío, entrada que no es un string o proveedor distinto de OpenAI.
- `ValidationError`: configuración inválida antes de llamar a la API, o respuesta
  que incumple el esquema después de dos intentos.
- `OutputParserException`: respuesta sin un objeto válido después de dos intentos.
- `LengthFinishReasonError`: respuesta truncada en ambos intentos.
- `OpenAIError`: error del SDK de OpenAI, como autenticación, cuota, red o timeout.
  Solo los fallos transitorios se reintentan según `LLM_MAX_RETRIES`.

Ejemplo de consumo asíncrono:

```python
import asyncio

from langchain_core.exceptions import OutputParserException
from openai import LengthFinishReasonError, OpenAIError
from pydantic import ValidationError

from chain import process_text


async def main():
    try:
        result = await process_text("FastAPI usa Redis; el servicio está caído.")
    except (ValidationError, OutputParserException, LengthFinishReasonError) as exc:
        print("Configuración o respuesta inválida:", type(exc).__name__)
    except ValueError as exc:
        print("Entrada o proveedor inválido:", exc)
    except OpenAIError as exc:
        print("Error de OpenAI:", type(exc).__name__)
    else:
        print(result.model_dump_json(indent=2))


asyncio.run(main())
```

El ejemplo contempla el agotamiento de los reintentos y los errores de
configuración. `ValidationError` se captura antes de `ValueError` porque es
una subclase de esta última. Los errores de autenticación no se reintentan.

## Pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

En Linux o macOS: `.venv/bin/python -m unittest discover -s tests -v`.
Las pruebas simulan HTTP sin credenciales reales. Incluyen recuperación ante
JSON mal formado, campos incompletos, truncamiento y texto ambiguo.
