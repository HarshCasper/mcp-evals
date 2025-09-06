import * as yaml from 'js-yaml';
import * as fs from 'fs';
import * as path from 'path';
import { openai } from "@ai-sdk/openai";
import { anthropic } from "@ai-sdk/anthropic";
import { type LanguageModel } from 'ai';
import { YamlEvalConfig, YamlEval, EvalConfig, EvalFunction } from './types.js';
import { grade } from './index.js';

/**
 * Load and parse a YAML configuration file
 */
export function loadYamlConfig(filePath: string): YamlEvalConfig {
  try {
    const absolutePath = path.resolve(filePath);
    const fileContents = fs.readFileSync(absolutePath, 'utf8');
    const config = yaml.load(fileContents) as YamlEvalConfig;
    
    if (!config || !config.evals || !Array.isArray(config.evals)) {
      throw new Error('Invalid YAML config: must have an "evals" array');
    }

    // Validate each eval
    for (const evalItem of config.evals) {
      if (!evalItem.name || !evalItem.description || !evalItem.prompt) {
        throw new Error(`Invalid eval: each eval must have "name", "description", and "prompt" fields`);
      }
    }

    return config;
  } catch (error) {
    if (error instanceof Error) {
      throw new Error(`Failed to load YAML config: ${error.message}`);
    }
    throw new Error('Failed to load YAML config: Unknown error');
  }
}

/**
 * Convert a YAML configuration to an EvalConfig
 */
export function yamlConfigToEvalConfig(yamlConfig: YamlEvalConfig, serverPath: string): EvalConfig {
  // Setup model
  let model: LanguageModel;
  if (yamlConfig.model) {
    if (yamlConfig.model.provider === 'openai') {
      // Set the API key as an environment variable if provided
      if (yamlConfig.model.api_key) {
        process.env.OPENAI_API_KEY = yamlConfig.model.api_key;
      }
      model = openai(yamlConfig.model.name as any);
    } else if (yamlConfig.model.provider === 'anthropic') {
      // Set the API key as an environment variable if provided
      if (yamlConfig.model.api_key) {
        process.env.ANTHROPIC_API_KEY = yamlConfig.model.api_key;
      }
      model = anthropic(yamlConfig.model.name as any);
    } else {
      throw new Error(`Unsupported model provider: ${yamlConfig.model.provider}`);
    }
  } else {
    // Default to GPT-4
    model = openai("gpt-4o");
  }

  // Convert YAML evals to EvalFunctions
  const evalFunctions: EvalFunction[] = yamlConfig.evals.map((yamlEval: YamlEval) => ({
    name: yamlEval.name,
    description: yamlEval.description,
    run: async (evalModel: LanguageModel) => {
      try {
        const result = await grade({
          model: evalModel,
          prompt: yamlEval.prompt,
          serverPath,
          systemPrompt: yamlConfig.grading_prompt
        });
        return JSON.parse(result);
      } catch (error) {
        // If JSON parsing fails, return a default structure
        return {
          accuracy: 0,
          completeness: 0,
          relevance: 0,
          clarity: 0,
          reasoning: 0,
          overall_comments: `Error running evaluation: ${error instanceof Error ? error.message : String(error)}`
        };
      }
    }
  }));

  return {
    model,
    evals: evalFunctions,
    env: yamlConfig.env
  };
}

/**
 * Load a YAML config file and convert it to an EvalConfig
 */
export function loadYamlEvalConfig(filePath: string, serverPath: string): EvalConfig {
  const yamlConfig = loadYamlConfig(filePath);
  return yamlConfigToEvalConfig(yamlConfig, serverPath);
}