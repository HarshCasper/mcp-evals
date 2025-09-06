import { type LanguageModel } from 'ai';

export interface EvalResult {
  accuracy: number;
  completeness: number;
  relevance: number;
  clarity: number;
  reasoning: number;
  overall_comments: string;
}

export interface EvalFunction {
  name: string;
  description: string;
  run: (model: LanguageModel) => Promise<EvalResult>;
}

export interface EvalConfig {
  model: LanguageModel;
  evals: EvalFunction[];
  env?: { [key: string]: string };
}

export interface YamlModelConfig {
  provider: 'openai' | 'anthropic';
  name: string;
  api_key?: string;
}

export interface YamlEval {
  name: string;
  description: string;
  prompt: string;
  expected_result?: string;
}

export interface YamlEvalConfig {
  model?: YamlModelConfig;
  evals: YamlEval[];
  env?: { [key: string]: string };
} 