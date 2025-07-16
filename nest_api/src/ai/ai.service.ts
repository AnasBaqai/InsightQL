import { Injectable, OnModuleInit } from '@nestjs/common';
import { OpenAI } from 'langchain/llms/openai';
import { ChatGroq } from '@langchain/groq';
import { createSqlAgent, SqlToolkit } from 'langchain/agents/toolkits/sql';
import { SqlDatabase } from 'langchain/sql_db';
import { DataSource } from 'typeorm';
import { RESULT } from './constants/results';
import { SQL_SUFFIX, SQL_PREFIX } from './constants/prompt';
import { InjectDataSource } from '@nestjs/typeorm';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import { QueryHistory } from './entities/chat.history.entity';
import { AiResponse } from './dto/ai-response.dto';
import { ChatHistoryResponseDto } from './dto/chatHistory-response.dto';

@Injectable()
export class AiService implements OnModuleInit {
  private executor: any;
  private model: any;
  private toolkit: SqlToolkit;

  constructor(
    @InjectDataSource('postgres') private postgresDataSource: DataSource,
    @InjectDataSource('sqlite') private sqliteDataSource: DataSource,
    @InjectModel('ChatHistory')
    private readonly chatHistoryModel: Model<QueryHistory>,
  ) {}

  async onModuleInit() {
    const postgresDb = await SqlDatabase.fromDataSourceParams({
      appDataSource: this.postgresDataSource,
    });

    const sqliteDb = await SqlDatabase.fromDataSourceParams({
      appDataSource: this.sqliteDataSource,
    });

    // Configure LLM based on environment variables
    const llmProvider = process.env.LLM_PROVIDER || 'groq'; // Default to Groq

    if (llmProvider === 'groq') {
      this.model = new ChatGroq({
        apiKey: process.env.GROQ_API_KEY,
        model: process.env.GROQ_MODEL || 'mixtral-8x7b-32768', // Better for structured tasks
        temperature: 0,
        maxTokens: 2048,
        streaming: false,
      });
    } else if (llmProvider === 'openai') {
      this.model = new OpenAI({
        openAIApiKey: process.env.OPENAI_API_KEY,
        temperature: 0,
      });
    } else {
      throw new Error(`Unsupported LLM provider: ${llmProvider}`);
    }

    this.toolkit = new SqlToolkit(postgresDb);

    this.executor = createSqlAgent(this.model, this.toolkit, {
      topK: 20,
      prefix: SQL_PREFIX,
      suffix: SQL_SUFFIX,
    });
  }

  async chat(prompt: string): Promise<AiResponse> {
    let aiResponse = new AiResponse();
    aiResponse.prompt = prompt; // Set the prompt early to avoid validation errors

    try {
      console.log('Starting chat with prompt:', prompt);
      console.log('LLM Provider:', process.env.LLM_PROVIDER);

      const result = await this.executor.call({ input: prompt });
      console.log('LLM execution completed');

      // Check if we got a valid result
      if (!result || !result.intermediateSteps) {
        console.log('No intermediate steps found in result');
        aiResponse.error = 'No response from LLM. Please try again.';
        return aiResponse;
      }

      console.log(
        'Number of intermediate steps:',
        result.intermediateSteps.length,
      );

      // Initialize with defaults to avoid validation errors
      aiResponse.sqlQuery = 'No SQL generated';
      aiResponse.result = [];

      result.intermediateSteps.forEach((step, index) => {
        console.log(`Step ${index}:`, step.action?.tool);
        if (step.action.tool === 'query-sql') {
          aiResponse.sqlQuery = step.action.toolInput;
          aiResponse.sqlQuery = aiResponse.sqlQuery
            .replace(/\\/g, '')
            .replace(/"/g, '');
          try {
            const observation = JSON.parse(step.observation);
            if (
              Array.isArray(observation) &&
              observation.every((obj) => typeof obj === 'object')
            ) {
              aiResponse.result = observation;
            }
          } catch (error) {
            console.log('Error parsing observation:', error);
          }
        }
      });

      // Only save to database if we have valid data
      if (aiResponse.sqlQuery && aiResponse.sqlQuery !== 'No SQL generated') {
        const chatHistory = new this.chatHistoryModel({
          prompt: aiResponse.prompt,
          sqlQuery: aiResponse.sqlQuery,
          queryResult: aiResponse.result,
        });

        await chatHistory.save();
        console.log('Chat history saved successfully');
      } else {
        console.log('Skipping database save - no valid SQL generated');
      }

      return aiResponse;
    } catch (e) {
      console.log('Error in chat method:', e);
      aiResponse.error = 'Server error. Try again with a different prompt.';
      return aiResponse;
    }
  }

  async getAllChatHistory(): Promise<Array<ChatHistoryResponseDto>> {
    const chatHistory: Array<QueryHistory> = await this.chatHistoryModel.find();

    if (!chatHistory) {
      return [];
    }

    const chatHistoryResponse: Array<ChatHistoryResponseDto> = chatHistory.map(
      (history) => {
        const chatHistoryResponseDto = new ChatHistoryResponseDto();
        chatHistoryResponseDto._id = history._id;
        chatHistoryResponseDto.prompt = history.prompt;
        chatHistoryResponseDto.sqlQuery = history.sqlQuery;
        chatHistoryResponseDto.result = history.queryResult;
        return chatHistoryResponseDto;
      },
    );
    return chatHistoryResponse;
  }
}
