// @ts-nocheck
// metrics.ts
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { NodeSDK } from '@opentelemetry/sdk-node';
import { resourceFromAttributes } from '@opentelemetry/resources';
import { SemanticResourceAttributes } from '@opentelemetry/semantic-conventions';
import { SimpleSpanProcessor } from '@opentelemetry/sdk-trace-base';
import { OTLPTraceExporter } from '@opentelemetry/exporter-trace-otlp-http';
import { trace, SpanStatusCode } from '@opentelemetry/api';
import { Registry, Counter, Histogram } from 'prom-client';
import express from 'express';

export interface MetricsConfig {
    metricsPort?: number;
    otelExporterUrl?: string;
    serviceName?: string;
    debug?: boolean;
    enableTracing?: boolean;
    otelEndpoint?: string;
  }

class Metrics {
  private static instance: Metrics;
  private sdk: NodeSDK;
  private tracer: any;
  private registry: Registry;
  private metrics: {
    toolCalls: Counter;
    toolErrors: Counter;
    toolLatency: Histogram;
  };
  private metricsServer: express.Application;

  private constructor(options = {}) {
    // Initialize OpenTelemetry if tracing is enabled
    if (options.enableTracing) {
        // Default to local collector endpoint if not specified
        const otelEndpoint = options.otelEndpoint || 'http://localhost:4318/v1/traces';
        console.log(`Configuring OpenTelemetry with endpoint: ${otelEndpoint}`);
        
        const exporterOptions = { 
            url: otelEndpoint,
            headers: {}, // Additional headers if needed
            concurrencyLimit: 10
        };
        
        this.sdk = new NodeSDK({
            resource: resourceFromAttributes({
                [SemanticResourceAttributes.SERVICE_NAME]: options.serviceName || 'mcp-server',
            }),
            spanProcessor: new SimpleSpanProcessor(new OTLPTraceExporter(exporterOptions)),
        });
        
        // Start the SDK
        try {
            this.sdk.start();
            console.log(`OpenTelemetry SDK started successfully, sending traces to ${otelEndpoint}`);
        } catch (error) {
            console.error('Failed to start OpenTelemetry SDK:', error);
        }
    } else {
        console.log('OpenTelemetry tracing is disabled');
    }
    
    this.tracer = trace.getTracer('mcp-tracer');

    // Initialize Prometheus metrics
    this.registry = new Registry();
    
    this.metrics = {
      toolCalls: new Counter({
        name: 'mcp_tool_calls_total',
        help: 'Total number of tool calls',
        labelNames: ['tool_name'],
        registers: [this.registry]
      }),
      toolErrors: new Counter({
        name: 'mcp_tool_errors_total',
        help: 'Total number of tool errors',
        labelNames: ['tool_name'],
        registers: [this.registry]
      }),
      toolLatency: new Histogram({
        name: 'mcp_tool_latency_seconds',
        help: 'Tool call latency in seconds',
        labelNames: ['tool_name'],
        buckets: [0.1, 0.5, 1, 2, 5],
        registers: [this.registry]
      })
    };

    // Setup metrics server
    this.metricsServer = express();
    this.metricsServer.get('/metrics', async (req, res) => {
      res.set('Content-Type', this.registry.contentType);
      res.end(await this.registry.metrics());
    });
  }

  static initialize(metricsPort = 9090, options = {}) {
    if (!Metrics.instance) {
      Metrics.instance = new Metrics(options);
      
      // Patch the McpServer prototype to add instrumentation
      const originalTool = McpServer.prototype.tool;
      McpServer.prototype.tool = function(...args) {
        // The tool method can be called with multiple signatures:
        // (name, description, parameters, handler)
        // (name, description, parameters, options, handler)
        const name = args[0];
        const description = args[1];
        const parameters = args[2];
        
        // Last arg is always the handler
        const handler = args[args.length - 1];
        
        // Check if options are present (second to last argument when length > 4)
        const hasOptions = args.length > 4;
        const options = hasOptions ? args[3] : undefined;
        
        // Create new args array with wrapped handler
        const newArgs = hasOptions 
            ? [name, description, parameters, options]
            : [name, description, parameters];
        
        // Add wrapped handler that includes instrumentation
        newArgs.push(async (...handlerArgs) => {
            const span = Metrics.instance.tracer.startSpan(`tool.${name}`);
            const startTime = process.hrtime();
            try {
                // Increment tool calls counter
                Metrics.instance.metrics.toolCalls.inc({ tool_name: name });
                const result = await handler(...handlerArgs);
                span.setStatus({ code: SpanStatusCode.OK });
                return result;
            }
            catch (error) {
                // Increment error counter
                Metrics.instance.metrics.toolErrors.inc({ tool_name: name });
                span.setStatus({
                    code: SpanStatusCode.ERROR,
                    message: error instanceof Error ? error.message : String(error)
                });
                throw error;
            }
            finally {
                // Record latency
                const [seconds, nanoseconds] = process.hrtime(startTime);
                const duration = seconds + nanoseconds / 1e9;
                Metrics.instance.metrics.toolLatency.observe({ tool_name: name }, duration);
                span.end();
            }
        });
        
        // Call original tool method with new args
        return originalTool.apply(this, newArgs);
      };

      // Start metrics server
      Metrics.instance.metricsServer.listen(metricsPort, () => {
        console.log(`Metrics server listening on port ${metricsPort}`);
      });
    }
    return Metrics.instance;
  }
}

export const metrics = {
  initialize: (metricsPort = 9090, options = {}) => Metrics.initialize(metricsPort, options)
};