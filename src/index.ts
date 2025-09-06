import { Client } from "@modelcontextprotocol/sdk/client/index.js";
import { Experimental_StdioMCPTransport } from "ai/mcp-stdio";
import {
  experimental_createMCPClient,
  streamText,
  type LanguageModel,
} from "ai";
import { openai } from "@ai-sdk/openai";
import { EvalConfig } from './types.js';
const defaultModel = openai("gpt-4o");

// Default grading system prompt used when a custom one is not provided
const DEFAULT_GRADING_PROMPT = `You are an expert evaluator assessing how well an LLM answers a given question. Review the provided answer and score it from 1 to 5 in each of the following categories:
        Accuracy – Does the answer contain factual errors or hallucinations?
        Completeness – Does the answer fully address all parts of the question?
        Relevance – Is the information directly related to the question?
        Clarity – Is the explanation easy to understand and well-structured?
        Reasoning – Does the answer show logical thinking or provide evidence or rationale?
        Return your evaluation as a JSON object in the format:
        {
            "accuracy": 1-5,
            "completeness": 1-5,
            "relevance": 1-5,
            "clarity": 1-5,
            "reasoning": 1-5,
            "overall_comments": "A short paragraph summarizing the strengths and weaknesses of the answer."
        }`;

export async function runEvals(model: LanguageModel=defaultModel, prompt: string,serverPath: string, env?: { [key: string]: string }) {
  const transport = new Experimental_StdioMCPTransport({
    command: "tsx",
    args: [serverPath],
    env: {
      ...Object.fromEntries(Object.entries(process.env).filter(([_, v]) => v !== undefined)),
      ...(env || {})
    } as Record<string, string>
  });

  const client = await experimental_createMCPClient({
    transport,
  });

  const tools = await client.tools();

  try {
    const result = streamText({
      model,
      tools,
      system:
        "You are an assistant responsible for evaluating the results of calling various tools. Given the user's query, use the tools available to you to answer the question.",
      prompt,
      maxRetries: 1,
      maxSteps: 10,
      onError: ({ error }) => {
        console.error(error);
      },
    });

    let fullText = '';
    for await (const chunk of result.fullStream) {
      if (chunk.type === 'text-delta') {
        fullText += chunk.textDelta;
      }
    }

    return fullText;
  } catch (error) {
    console.error('Error in runEvals:', error);
    throw error;
  }
}

// New options-first signature with backward compatibility for the old positional signature
export async function grade(options: {
  model?: LanguageModel;
  prompt: string;
  serverPath?: string;
  systemPrompt?: string;
}): Promise<string>;
export async function grade(model: LanguageModel, prompt: string, serverPath?: string): Promise<string>;
export async function grade(arg1: any, arg2?: any, arg3?: any): Promise<string> {
  // Determine if called with options object or legacy signature
  const isOptionsCall = typeof arg1 === 'object' && arg1 !== null && 'prompt' in arg1;

  const model: LanguageModel = isOptionsCall ? (arg1.model ?? defaultModel) : (arg1 as LanguageModel) ?? defaultModel;
  const prompt: string = isOptionsCall ? arg1.prompt : (arg2 as string);
  const serverPath: string | undefined = isOptionsCall ? arg1.serverPath : (arg3 as string | undefined);
  const systemPrompt: string = isOptionsCall ? (arg1.systemPrompt ?? DEFAULT_GRADING_PROMPT) : DEFAULT_GRADING_PROMPT;

  const finalServerPath = serverPath || process.argv[3];
  if (!finalServerPath) {
    throw new Error('Server path not provided');
  }

  const result = await runEvals(model, prompt, finalServerPath);
  const evalPromt = `Here is the user input: ${prompt}\nHere is the LLM's answer: ${result}`;
  const evalResult = streamText({
    model,
    maxRetries: 1,
    maxSteps: 10,
    system: systemPrompt,
    prompt: evalPromt,
    onError: ({ error }) => {
      console.error(error);
    },
  });

  for await (const _ of (evalResult as any).fullStream) {
  }

  return await (evalResult as any).text;
}

export async function runAllEvals(config: EvalConfig, serverPath: string) {
  const results = new Map<string, any>();
  let transport;
  
  try {
    transport = new Experimental_StdioMCPTransport({
      command: "tsx",
      args: [serverPath],
      env: {
        ...Object.fromEntries(Object.entries(process.env).filter(([_, v]) => v !== undefined)),
        ...(config.env || {})
      } as Record<string, string>
    });

    const client = await experimental_createMCPClient({
      transport,
    });

    for (const evaluation of config.evals) {
      console.log(`Running ${evaluation.name}...`);
      try {
        const result = await evaluation.run(config.model);
        results.set(evaluation.name, result);
      } catch (error) {
        console.error(`Error running ${evaluation.name}:`, error);
        results.set(evaluation.name, { error: error instanceof Error ? error.message : String(error) });
      }
    }
    
    return results;
  } finally {
    // Clean up the transport
    if (transport) {
      await transport.close?.();
    }
  }
}

// Export everything needed by consumers
export * from './types.js';
export { metrics } from './metrics.js';
export type { MetricsConfig } from './metrics.js';