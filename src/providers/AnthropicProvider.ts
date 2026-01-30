import Anthropic from '@anthropic-ai/sdk';
import {
  APIError,
  RateLimitError,
  AuthenticationError,
  BadRequestError,
  APIConnectionError,
  APIConnectionTimeoutError
} from '@anthropic-ai/sdk/error';
import { AIProvider, AIProviderConfig, ReviewRequest, ReviewResponse } from './AIProvider';
import * as core from '@actions/core';
import { baseCodeReviewPrompt, updateReviewPrompt } from '../prompts';
import { ContentBlock, TextBlock } from '@anthropic-ai/sdk/resources';

export class AnthropicProvider implements AIProvider {
  private config!: AIProviderConfig;
  private client!: Anthropic;

  async initialize(config: AIProviderConfig): Promise<void> {
    this.config = config;
    this.client = new Anthropic({
      apiKey: config.apiKey
    });
    core.info(`AnthropicProvider initialized with model: ${config.model}`);
  }

  async review(request: ReviewRequest): Promise<ReviewResponse> {
    core.info('=== Starting Anthropic API Request ===');
    core.debug(`Request structure: ${JSON.stringify(request, null, 2)}`);

    const systemPrompt = this.buildSystemPrompt(request);
    const userPrompt = this.buildPullRequestPrompt(request);

    core.info(`System prompt length: ${systemPrompt.length} characters`);
    core.info(`User prompt length: ${userPrompt.length} characters`);
    core.debug(`System prompt: ${systemPrompt}`);
    core.debug(`User prompt (first 500 chars): ${userPrompt.substring(0, 500)}...`);

    try {
      const response = await this.client.messages.create({
        model: this.config.model,
        max_tokens: this.config.maxTokens ?? 4000,
        system: systemPrompt,
        messages: [
          {
            role: 'user',
            content: userPrompt,
          },
        ],
        temperature: this.config.temperature ?? 0.3,
        // Use output_config to guarantee valid JSON responses without markdown formatting
        output_config: {
          format: {
            type: 'json_schema',
            schema: {
              type: 'object',
              properties: {
                summary: { type: 'string' },
                comments: {
                  type: 'array',
                  items: {
                    type: 'object',
                    properties: {
                      path: { type: 'string' },
                      line: { type: 'number' },
                      comment: { type: 'string' }
                    },
                    required: ['path', 'line', 'comment'],
                    additionalProperties: false
                  }
                },
                suggestedAction: {
                  type: 'string',
                  enum: ['APPROVE', 'REQUEST_CHANGES', 'COMMENT']
                },
                confidence: { type: 'number' }
              },
              required: ['summary', 'comments', 'suggestedAction', 'confidence'],
              additionalProperties: false
            }
          }
        }
      });

      core.info('=== Anthropic API Response Received ===');
      core.info(`Response ID: ${response.id}`);
      core.info(`Model: ${response.model}`);
      core.info(`Stop reason: ${response.stop_reason}`);
      core.info(`Input tokens: ${response.usage.input_tokens}`);
      core.info(`Output tokens: ${response.usage.output_tokens}`);

      // Log cache token usage if available (prompt caching feature)
      if (response.usage.cache_creation_input_tokens) {
        core.info(`Cache creation tokens: ${response.usage.cache_creation_input_tokens}`);
      }
      if (response.usage.cache_read_input_tokens) {
        core.info(`Cache read tokens: ${response.usage.cache_read_input_tokens}`);
      }

      core.info(`Content blocks: ${response.content.length}`);

      // Log the full response structure
      core.debug(`Full response object: ${JSON.stringify(response, null, 2)}`);

      // Log each content block
      response.content.forEach((block, index) => {
        core.info(`Content block ${index}: type=${block.type}`);
        if (block.type === 'text') {
          const textBlock = block as TextBlock;
          core.info(`Text block ${index} length: ${textBlock.text.length} characters`);
          core.debug(`Text block ${index} content: ${textBlock.text}`);
        }
      });

      const parsedResponse = this.parseResponse(response);
      core.info('=== Parsed Response ===');
      core.info(`Summary length: ${parsedResponse.summary.length}`);
      core.info(`Line comments: ${parsedResponse.lineComments?.length ?? 0}`);
      core.info(`Suggested action: ${parsedResponse.suggestedAction}`);
      core.info(`Confidence: ${parsedResponse.confidence}`);
      core.debug(`Full parsed response: ${JSON.stringify(parsedResponse, null, 2)}`);

      return parsedResponse;
    } catch (error) {
      core.error('=== Anthropic API Error ===');

      // Handle specific Anthropic SDK errors
      if (error instanceof RateLimitError) {
        const retryAfter = error.headers?.get('retry-after');
        core.error(`Rate limit exceeded. ${retryAfter ? `Retry after: ${retryAfter}` : 'Please retry later.'}`);
        core.error(`Status: ${error.status}`);
        core.error(`Request ID: ${error.requestID}`);
      } else if (error instanceof AuthenticationError) {
        core.error('Authentication failed. Please check your API key.');
        core.error(`Status: ${error.status}`);
      } else if (error instanceof BadRequestError) {
        core.error('Invalid request parameters.');
        core.error(`Status: ${error.status}`);
        core.error(`Error details: ${JSON.stringify(error.error)}`);
      } else if (error instanceof APIConnectionTimeoutError) {
        core.error('Request timed out. The API took too long to respond.');
      } else if (error instanceof APIConnectionError) {
        core.error('Network connection error. Please check your internet connection.');
        core.error(`Cause: ${error.message}`);
      } else if (error instanceof APIError) {
        core.error(`API Error occurred.`);
        core.error(`Status: ${error.status}`);
        core.error(`Request ID: ${error.requestID}`);
        core.error(`Error details: ${JSON.stringify(error.error)}`);
      } else if (error instanceof Error) {
        core.error(`Error name: ${error.name}`);
        core.error(`Error message: ${error.message}`);
        core.error(`Error stack: ${error.stack}`);
      } else {
        core.error(`Unknown error: ${JSON.stringify(error)}`);
      }

      throw error;
    }
  }

  private buildPullRequestPrompt(request: ReviewRequest): string {
    return JSON.stringify({
      type: 'code_review',
      files: request.files,
      pr: request.pullRequest,
      context: request.context,
      previousReviews: request.previousReviews?.map(review => ({
        summary: review.summary,
        lineComments: review.lineComments.map(comment => ({
          path: comment.path,
          line: comment.line,
          comment: comment.comment
        }))
      }))
    });
  }

  private buildSystemPrompt(request: ReviewRequest): string {
    const isUpdate = request.context.isUpdate;

    return `
      ${baseCodeReviewPrompt}
      ${isUpdate ? updateReviewPrompt : ''}
    `.trim();
  }

  private parseResponse(response: Anthropic.Message): ReviewResponse {
    core.info('=== Starting Response Parsing ===');

    try {
      // Safely extract text content
      const textContent = this.extractTextContent(response.content);

      if (!textContent) {
        core.error('No text content found in response');
        throw new Error('No text content in response');
      }

      core.info(`Extracted text content length: ${textContent.length} characters`);
      core.debug(`Raw text content: ${textContent}`);

      let text = textContent;

      // With output_config.format set to json_schema, the response should be valid JSON
      // without markdown formatting. However, we keep this as a safety fallback.
      const codeBlockMatch = text.match(/^```(?:json)?\s*\n?([\s\S]*?)\n?```$/);
      if (codeBlockMatch) {
        text = codeBlockMatch[1];
        core.warning('Unexpected markdown code blocks found - this should not happen with json_schema output_config');
        core.debug(`Text after stripping code blocks: ${text}`);
      }

      core.info('Attempting to parse JSON...');
      const content = JSON.parse(text.trim());

      core.info('JSON parsing successful');
      core.debug(`Parsed JSON object: ${JSON.stringify(content, null, 2)}`);

      const result: ReviewResponse = {
        summary: content.summary || 'No summary provided',
        lineComments: content.comments || [],
        suggestedAction: content.suggestedAction || 'COMMENT',
        confidence: content.confidence ?? 0,
      };

      core.info(`Created ReviewResponse with ${result.lineComments?.length ?? 0} line comments`);

      return result;
    } catch (error) {
      core.error('=== Response Parsing Failed ===');
      core.error(`Parse error: ${error}`);

      // Log the actual response for debugging
      try {
        const textContent = this.extractTextContent(response.content);
        if (textContent) {
          core.error('--- Response content that failed to parse (first 1000 chars) ---');
          core.error(textContent.substring(0, 1000));
          if (textContent.length > 1000) {
            core.error(`... (${textContent.length - 1000} more characters)`);
          }
        }
      } catch (e) {
        core.error(`Could not extract response content for debugging: ${e}`);
      }

      return {
        summary: 'Failed to parse AI response',
        lineComments: [],
        suggestedAction: 'COMMENT',
        confidence: 0,
      };
    }
  }

  /**
   * Safely extracts text content from response blocks
   */
  private extractTextContent(content: ContentBlock[]): string | null {
    if (!content || content.length === 0) {
      core.warning('Response content array is empty or null');
      return null;
    }

    core.info(`Extracting text from ${content.length} content block(s)`);

    // Find the first text block
    const textBlock = content.find(block => block.type === 'text') as TextBlock | undefined;

    if (!textBlock) {
      core.warning(`No text block found. Available block types: ${content.map(b => b.type).join(', ')}`);
      return null;
    }

    core.info('Text block found successfully');
    return textBlock.text;
  }
}
