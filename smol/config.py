import os

from openinference.instrumentation.smolagents import SmolagentsInstrumentor
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from smolagents import LiteLLMModel
from smolagents import OpenAIServerModel


endpoint = "http://0.0.0.0:6006/v1/traces"
trace_provider = TracerProvider()
trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)

match os.getenv("MODEL"):
  case "gpt4o":
    model = OpenAIServerModel("gpt-4o", api_key=os.environ["OPENAI_API_KEY"])
  case "claude":
    model = LiteLLMModel(model_id="anthropic/claude-3-5-sonnet-latest", api_key=os.environ["ANTHROPIC_API_KEY"])
  case _:
    model = OpenAIServerModel("default", api_base="http://localhost:3000/v1", api_key="[EMPTY]", tool_choice="auto")
