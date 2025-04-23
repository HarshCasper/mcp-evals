import { grade } from "../../src/index"
import { openai } from "@ai-sdk/openai"
import * as dotenv from 'dotenv'
import type { EvalFunction } from '../../src/types'

// Load .env file
dotenv.config()

export const weatherEval: EvalFunction = {
  name: 'Weather Tool Evaluation',
  description: 'Evaluates the accuracy and completeness of weather information retrieval',
  run: async () => {
    const result = await grade(openai("gpt-4"), "What is the weather in New York?");
    return JSON.parse(result);
  }
};
