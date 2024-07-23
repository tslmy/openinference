import {
  ChatCompletionsFunctionToolCall,
  ChatCompletionsToolCallUnion,
  ChatMessageContentItemUnion,
  ChatMessageImageContentItem,
  ChatMessageTextContentItem,
  ChatRequestAssistantMessage,
  ChatRequestFunctionMessage,
  ChatRequestMessageUnion,
  ChatRequestSystemMessage,
  ChatRequestToolMessage,
  ChatRequestUserMessage,
} from "@azure/openai";

/**
 * Utility function that uses the type system to check if a switch statement is exhaustive.
 * If the switch statement is not exhaustive, there will be a type error caught in typescript
 *
 * See https://stackoverflow.com/questions/39419170/how-do-i-check-that-a-switch-block-is-exhaustive-in-typescript for more details.
 */
export function assertUnreachable(_: never): never {
  throw new Error("Unreachable");
}

export function isString(value: unknown): value is string {
  return typeof value === "string";
}

type MessageWithContent =
  | ChatRequestSystemMessage
  | ChatRequestUserMessage
  | ChatRequestAssistantMessage
  | ChatRequestToolMessage
  | ChatRequestFunctionMessage;

/**
 * Type guard to check if a message has content
 * @param message - The message to check
 * @returns true if the message has content, false otherwise
 */
export function isMessageWithContent(
  message: ChatRequestMessageUnion,
): message is MessageWithContent {
  return (message as MessageWithContent).content != null;
}

/**
 * Type guard to check if message content is an image (i.e., a ChatMessageImageContentItem)
 * @param messageContent - The message content to check
 * @returns true if the message content is an image, false otherwise
 */
export function isChatMessageImageContent(
  messageContent: ChatMessageContentItemUnion,
): messageContent is ChatMessageImageContentItem {
  return (messageContent as ChatMessageImageContentItem).type === "image_url";
}

/**
 * Type guard to check if message content is text (i.e., a ChatMessageTextContentItem)
 * @param messageContent
 * @returns true if the message content is text, false otherwise
 */
export function isChatMessageTextContent(
  messageContent: ChatMessageContentItemUnion,
): messageContent is ChatMessageTextContentItem {
  return (messageContent as ChatMessageTextContentItem).type === "text";
}

/**
 * Type guard to check if the tool call is a function tool call
 * @param toolCall - The tool call to check
 * @returns true if the tool call is a function tool call, false otherwise
 */
export function isFunctionToolCall(
  toolCall: ChatCompletionsToolCallUnion,
): toolCall is ChatCompletionsFunctionToolCall {
  return toolCall.type === "function";
}
