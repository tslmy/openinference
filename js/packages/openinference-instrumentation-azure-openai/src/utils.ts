import {
  SemanticConventions,
  MESSAGE_CONTENT_TYPE,
  MESSAGE_CONTENTS,
  MimeType,
} from "@arizeai/openinference-semantic-conventions";
import {
  ChatCompletions,
  ChatMessageContentItemUnion,
  ChatRequestMessageUnion,
  ChatResponseMessage,
  Completions,
  Embeddings,
} from "@azure/openai";
import { Attributes } from "@opentelemetry/api";
import {
  assertUnreachable,
  isChatMessageImageContent,
  isChatMessageTextContent,
  isFunctionToolCall,
  isMessageWithContent,
} from "./typeUtils";

function getChatMessageContentAttributes(
  content: ChatMessageContentItemUnion,
): Attributes {
  if (isChatMessageTextContent(content)) {
    return {
      [MESSAGE_CONTENT_TYPE]: content.type,
      [SemanticConventions.MESSAGE_CONTENT]: content.text,
    };
  }
  if (isChatMessageImageContent(content)) {
    return {
      [MESSAGE_CONTENT_TYPE]: content.type,
      [SemanticConventions.MESSAGE_CONTENT]: content.imageUrl.url,
    };
  }
  return {};
}

function getChatCompletionInputMessageAttributes(
  message: ChatRequestMessageUnion,
): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };

  if (!isMessageWithContent(message)) {
    return attributes;
  }
  if (typeof message.content === "string") {
    attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
  }

  switch (message.role) {
    case "user":
      // String contents are captured above
      if (typeof message.content === "string") {
        break;
      }
      message.content.forEach((content, index) => {
        const indexPrefix = `${MESSAGE_CONTENTS}.${index}.`;
        const contentAttributes = getChatMessageContentAttributes(content);
        for (const [key, value] of Object.entries(contentAttributes)) {
          attributes[`${indexPrefix}${key}`] = value;
        }
      });
      break;
    case "assistant":
      if (message.toolCalls) {
        message.toolCalls.forEach((toolCall, index) => {
          // Make sure the tool call has a function
          if (isFunctionToolCall(toolCall)) {
            const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}.`;
            attributes[
              toolCallIndexPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME
            ] = toolCall.function.name;
            attributes[
              toolCallIndexPrefix +
                SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
            ] = toolCall.function.arguments;
          }
        });
      }
      break;
    case "function":
      attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME] = message.name;
      break;
    case "tool":
      // There's nothing to add for the tool. There is a toolCallId, but there are no
      // semantic conventions for it
      break;
    case "system":
      // There's nothing to add for the system. String content is captured above
      break;
    default:
      assertUnreachable(message);
  }
  return attributes;
}

/**
 * Converts the messages of a chat completions request to LLM input messages
 */
export function getLLMInputMessagesAttributes(
  messages: ChatRequestMessageUnion[],
): Attributes {
  return messages.reduce((acc, message, index) => {
    const messageAttributes = getChatCompletionInputMessageAttributes(message);
    const indexPrefix = `${SemanticConventions.LLM_INPUT_MESSAGES}.${index}.`;
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {} as Attributes);
}

function getChatCompletionOutputMessageAttributes(
  message: ChatResponseMessage,
): Attributes {
  const role = message.role;
  const attributes: Attributes = {
    [SemanticConventions.MESSAGE_ROLE]: role,
  };
  if (message.content) {
    attributes[SemanticConventions.MESSAGE_CONTENT] = message.content;
  }
  if (message.toolCalls) {
    message.toolCalls.forEach((toolCall, index) => {
      const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}.`;
      // Double check that the tool call has a function
      // NB: OpenAI only supports tool calls with functions right now but this may change
      if (isFunctionToolCall(toolCall)) {
        attributes[
          toolCallIndexPrefix + SemanticConventions.TOOL_CALL_FUNCTION_NAME
        ] = toolCall.function.name;
        attributes[
          toolCallIndexPrefix +
            SemanticConventions.TOOL_CALL_FUNCTION_ARGUMENTS_JSON
        ] = toolCall.function.arguments;
      }
    });
  }
  if (message.functionCall) {
    attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME] =
      message.functionCall.name;
    attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON] =
      message.functionCall.arguments;
  }
  return attributes;
}

/**
 * Converts the chat completion result to LLM output attributes
 */
export function getChatCompletionLLMOutputMessagesAttributes(
  chatCompletion: ChatCompletions,
): Attributes {
  // Right now support just the first choice
  const choice = chatCompletion.choices[0];
  if (!choice || choice.message == null) {
    return {};
  }
  return [choice.message].reduce((acc, message, index) => {
    const indexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.${index}.`;
    const messageAttributes = getChatCompletionOutputMessageAttributes(message);
    // Flatten the attributes on the index prefix
    for (const [key, value] of Object.entries(messageAttributes)) {
      acc[`${indexPrefix}${key}`] = value;
    }
    return acc;
  }, {} as Attributes);
}

/**
 * Get usage attributes
 * @param completion - The completion to get usage attributes from
 * @return The usage attributes
 */
export function getUsageAttributes(
  completion: ChatCompletions | Completions,
): Attributes {
  if (completion.usage) {
    return {
      [SemanticConventions.LLM_TOKEN_COUNT_COMPLETION]:
        completion.usage.completionTokens,
      [SemanticConventions.LLM_TOKEN_COUNT_PROMPT]:
        completion.usage.promptTokens,
      [SemanticConventions.LLM_TOKEN_COUNT_TOTAL]: completion.usage.totalTokens,
    };
  }
  return {};
}

/**
 * Converts the body of a completions request to input attributes
 * @param prompts - The prompts of the completions request
 * @return The input attributes
 */
export function getCompletionInputValueAndMimeType(
  prompts: string[],
): Attributes {
  const prompt = prompts[0]; // Only single prompts are currently supported
  if (prompt === undefined) {
    return {};
  }
  return {
    [SemanticConventions.INPUT_VALUE]: prompt,
    [SemanticConventions.INPUT_MIME_TYPE]: MimeType.TEXT,
  };
}

/**
 * Converts the completion result to output attributes
 * @param completion - The completion to get output attributes from
 * @return The output attributes
 */
export function getCompletionOutputValueAndMimeType(
  completion: Completions,
): Attributes {
  // Right now support just the first choice
  const choice = completion.choices[0];
  if (!choice) {
    return {};
  }
  return {
    [SemanticConventions.OUTPUT_VALUE]: String(choice.text),
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
  };
}

/**
 * Converts the embedding result payload to embedding text attributes
 * @param input - The input text to get embedding attributes from
 * @return The embedding text attributes
 */
export function getEmbeddingTextAttributes(input: string[]): Attributes {
  return input.reduce((acc, input, index) => {
    const indexPrefix = `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.`;
    acc[`${indexPrefix}${SemanticConventions.EMBEDDING_TEXT}`] = input;
    return acc;
  }, {} as Attributes);
}

/**
 * Converts the embedding result payload to embedding vector attributes
 * @param embeddings - The embeddings to get attributes from
 * @return The embedding vector attributes
 */
export function getEmbeddingVectorAttributes(
  embeddings: Embeddings,
): Attributes {
  return embeddings.data.reduce((acc, embedding, index) => {
    const indexPrefix = `${SemanticConventions.EMBEDDING_EMBEDDINGS}.${index}.`;
    acc[`${indexPrefix}${SemanticConventions.EMBEDDING_VECTOR}`] =
      embedding.embedding;
    return acc;
  }, {} as Attributes);
}
