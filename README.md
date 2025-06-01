# 🧠 AI English Coach for Software Engineers

Un asistente conversacional inteligente que ayuda a profesionales de TI de habla hispana a mejorar su inglés técnico, practicar entrevistas y desarrollar habilidades sociales. Entrenado con datos sintéticos y reales, el modelo responde en tres partes: corrección gramatical, respuesta experta y continuación de la conversación.

---

## 🚀 Características

- 🎤 Reconocimiento de voz (Whisper) con Voice Activity Detection
- 🤖 Modelo fine-tuned con LLaMA 3.1 8B + Unsloth + LoRA
- 💬 Respuestas estructuradas: mejora de inglés, respuesta técnica, y seguimiento natural
- 🧪 Evaluación automática: GPT-4o como juez + embeddings para análisis semántico
- 🎙️ Salida por voz con KokoroTTS o ElevenLabs
- 🧍 Animación VTuber en tiempo real (opcional)
- 🌎 Interfaz en vivo para simulaciones de entrevistas o clases interactivas

---

## 📦 Estructura del Proyecto

```

.
├── backend/
│   └── main.py        # API Flask con endpoint /generate
├── datasets/
│   └── curated/       # Datos reales y sintéticos para entrenamiento
├── frontend/
│   └── app.js         # Voz, transcripción y TTS
├── models/
│   └── master\_model/  # Modelo fine-tuned con LoRA
├── notebooks/
│   └── evaluation/    # Métricas automáticas y visualizaciones
└── README.md

```

---

## 🔧 Instalación

```bash
git clone https://github.com/tu_usuario/ai-english-coach.git
cd ai-english-coach
poetry install
```

Configura tu archivo `.env` con la clave de OpenAI y otros valores:

```env
OPENAI_API_KEY=sk-...
TTS_PROVIDER=kokoro
```

---

## 🧠 Entrenamiento

El modelo se entrena usando Unsloth + LoRA con partición:

* 80% train
* 10% val
* 10% test

Y un enfoque de enmascaramiento para aprender únicamente las respuestas del asistente.

---

## 🧪 Evaluación

Se aplican dos métricas principales:

| Dimensión             | Métrica                        | Método                   |
| --------------------- | ------------------------------ | ------------------------ |
| Precisión técnica     | LLM-as-grader (GPT-4o)         | Judge Score (1–10)       |
| Similaridad semántica | Embeddings + Cosine Similarity | `text-embedding-3-small` |

Visualización de métricas en Plotly (gauge charts) tras ejecución en `evaluation.ipynb`.

---

## 🧠 API de Backend

### POST `/generate`

Envía una conversación (formato estilo ShareGPT) y recibe una respuesta generada estructurada.

**Request**

```json
[
  {"role": "system", "content": "..."},
  {"role": "user", "content": "Tell me about databases."}
]
```

**Response**

```json
{
  "enhancement": "Let me tell you about databases...",
  "expert_answer": "...",
  "continue": "Would you like to know about SQL or NoSQL?"
}
```

---

## 🎯 Casos de uso

* Simulación de entrevistas técnicas
* Mejora del inglés conversacional profesional
* Integración en bootcamps o cursos de inglés para TI
* Entrenamiento en habilidades blandas (soft skills)

---

## 📈 Resultados esperados

* ↑ Precisión gramatical y técnica en inglés
* ↑ Confianza para entrevistas en inglés
* ↑ Posibilidad de aumentar ingresos hasta 30% (según estudios ENI, 2023)

---

## 🧠 System Prompt

```text
Eres un asistente experto en inglés profesional y tecnología.
Siempre responde en XML estructurado con:
<response>
  <enhancement>...</enhancement>
  <expert-answer>...</expert-answer>
  <continue>...</continue>
</response>
```

---

## ⚖️ Consideraciones Éticas

* No se almacenan conversaciones sin consentimiento explícito
* El modelo puede fallar: siempre validar respuestas críticas
* Se evita sesgo de género o cultural al curar el dataset

---

## 📌 Próximos pasos

* Fine-tuning específico por dominio (DevOps, Frontend, Data)
* Feedback por voz (con análisis fonético)
* Versión offline lightweight (GGUF + llama.cpp)

---

## 🧪 Demo sugerida (5 prompts)

1. "How would you explain what a REST API is?"
2. "Can you correct my sentence: 'I have many experience in backend.'"
3. "Simulate an interview for a software engineer position."
4. "What are the SOLID principles?"
5. "Give me feedback on my English: 'I am study very hard the programming.'"

---

## 🧑‍💻 Créditos

Proyecto de titulación — Maestría en Inteligencia Artificial Aplicada
Autor: Danny
Asistencia: LLaMA 3.1 8B + Unsloth + GPT-4o

