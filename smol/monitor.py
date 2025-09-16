try:
  from openinference.instrumentation.smolagents import SmolagentsInstrumentor
  from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
  from opentelemetry.sdk.trace import TracerProvider
  from opentelemetry.sdk.trace.export import SimpleSpanProcessor
except ImportError:
  pass
else:
  endpoint = "http://0.0.0.0:6006/v1/traces"
  trace_provider = TracerProvider()
  trace_provider.add_span_processor(SimpleSpanProcessor(OTLPSpanExporter(endpoint)))

  SmolagentsInstrumentor().instrument(tracer_provider=trace_provider)
