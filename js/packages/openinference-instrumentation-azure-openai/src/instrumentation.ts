import type * as AzureOpenAI from "@azure/openai";
import {
  InstrumentationBase,
  InstrumentationConfig,
  InstrumentationModuleDefinition,
  InstrumentationNodeModuleDefinition,
  safeExecuteInTheMiddle,
} from "@opentelemetry/instrumentation";
import {
  diag,
  context,
  trace,
  SpanKind,
  Attributes,
  SpanStatusCode,
  Span,
} from "@opentelemetry/api";
import { VERSION } from "./version";
import {
  SemanticConventions,
  OpenInferenceSpanKind,
  MimeType,
  MESSAGE_CONTENT,
  MESSAGE_CONTENT_TYPE,
} from "@arizeai/openinference-semantic-conventions";
import {
  ChatRequestMessageUnion,
  GetChatCompletionsOptions,
  ChatCompletions,
  ChatRequestMessage,
  ChatRequestSystemMessage,
  ChatRequestUserMessage,
  ChatRequestAssistantMessage,
  ChatRequestToolMessage,
  ChatRequestFunctionMessage,
  ChatMessageContentItemUnion,
  ChatMessageImageContentItem,
  ChatMessageTextContentItem,
  Completions,
} from "@azure/openai";
import { assertUnreachable, isString } from "./typeUtils";
import { isTracingSuppressed } from "@opentelemetry/core";
import {
  getChatCompletionLLMOutputMessagesAttributes,
  getCompletionInputValueAndMimeType,
  getCompletionOutputValueAndMimeType,
  getEmbeddingTextAttributes,
  getEmbeddingVectorAttributes,
  getLLMInputMessagesAttributes,
  getUsageAttributes,
} from "./utils";

const MODULE_NAME = "@azure/openai";

/**
 * Flag to check if the openai module has been patched
 * Note: This is a fallback in case the module is made immutable (e.x. Deno, webpack, etc.)
 */
let _isOpenInferencePatched = false;

/**
 * function to check if instrumentation is enabled / disabled
 */
export function isPatched() {
  return _isOpenInferencePatched;
}

/**
 * Resolves the execution context for the current span
 * If tracing is suppressed, the span is dropped and the current context is returned
 * @param span
 */
function getExecContext(span: Span) {
  const activeContext = context.active();
  const suppressTracing = isTracingSuppressed(activeContext);
  const execContext = suppressTracing
    ? trace.setSpan(context.active(), span)
    : activeContext;
  // Drop the span from the context
  if (suppressTracing) {
    trace.deleteSpan(activeContext);
  }
  return execContext;
}
export class OpenAIInstrumentation extends InstrumentationBase<
  typeof AzureOpenAI
> {
  constructor(config?: InstrumentationConfig) {
    super(
      "@arizeai/openinference-instrumentation-azure-openai",
      VERSION,
      Object.assign({}, config),
    );
  }

  protected init(): InstrumentationModuleDefinition<typeof AzureOpenAI> {
    const module = new InstrumentationNodeModuleDefinition<typeof AzureOpenAI>(
      "@azure/openai",
      ["^1.0.0-beta.12"],
      this.patch.bind(this),
      this.unpatch.bind(this),
    );
    return module;
  }

  /**
   * Manually instruments the OpenAI module. This is needed when the module is not loaded via require (commonjs)
   * @param {openai} module
   */
  manuallyInstrument(module: typeof AzureOpenAI) {
    diag.debug(`Manually instrumenting ${MODULE_NAME}`);
    this.patch(module);
  }

  /**
   * Patches the OpenAI module
   */
  private patch(
    module: typeof AzureOpenAI & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Applying patch for ${MODULE_NAME}@${moduleVersion}`);
    if (module?.openInferencePatched || _isOpenInferencePatched) {
      return module;
    }
    // eslint-disable-next-line @typescript-eslint/no-this-alias
    const instrumentation: OpenAIInstrumentation = this;

    type ChatCompletionCreateType =
      typeof module.OpenAIClient.prototype.getChatCompletions;

    this._wrap(
      module.OpenAIClient.prototype,
      "getChatCompletions",
      (original: ChatCompletionCreateType) => {
        return function patchedGetChatCompletions(
          this: unknown,
          ...args: Parameters<ChatCompletionCreateType>
        ) {
          const deploymentName = args[0];
          const messages = args[1];
          const options = args[2];
          const span = instrumentation.tracer.startSpan(
            `OpenAI Chat Completions`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: deploymentName,
                [SemanticConventions.INPUT_VALUE]: JSON.stringify(messages),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(options),
                ...getLLMInputMessagesAttributes(messages),
              },
            },
          );

          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle<
            ReturnType<ChatCompletionCreateType>
          >(
            () => {
              return context.with(execContext, () => {
                return original.apply(this, args);
              });
            },
            (error) => {
              // Push the error to the span
              if (error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            // Record the results
            span.setAttributes({
              [SemanticConventions.OUTPUT_VALUE]: JSON.stringify(result),
              [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.JSON,
              ...getChatCompletionLLMOutputMessagesAttributes(result),
              ...getUsageAttributes(result),
            });
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
            return result;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    this._wrap(
      module.OpenAIClient.prototype,
      "streamChatCompletions",
      (original) => {
        return function patchedStreamChatCompletions(this: unknown, ...args) {
          const deploymentName = args[0];
          const messages = args[1];
          const options = args[2];
          const span = instrumentation.tracer.startSpan(
            `Azure OpenAI Stream Chat Completions`,
            {
              kind: SpanKind.INTERNAL,
              attributes: {
                [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                  OpenInferenceSpanKind.LLM,
                [SemanticConventions.LLM_MODEL_NAME]: deploymentName,
                [SemanticConventions.INPUT_VALUE]: JSON.stringify(messages),
                [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
                [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
                  JSON.stringify(options),
                ...getLLMInputMessagesAttributes(messages),
              },
            },
          );
          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle(
            () => {
              return context.with(execContext, () => {
                return original.apply(this, args);
              });
            },
            (error) => {
              if (error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            const [leftStream, rightStream] = result;
            span.setAttributes({
              ...getCompletionOutputValueAndMimeType(result),
              ...getUsageAttributes(result),
            });
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
            return result;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    this._wrap(module.OpenAIClient.prototype, "getCompletions", (original) => {
      return function patchedGetCompletions(this: unknown, ...args) {
        const deploymentName = args[0];
        const prompts = args[1];
        const options = args[2];
        const span = instrumentation.tracer.startSpan(`OpenAI Completions`, {
          kind: SpanKind.INTERNAL,
          attributes: {
            [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
              OpenInferenceSpanKind.LLM,
            [SemanticConventions.LLM_MODEL_NAME]: deploymentName,
            [SemanticConventions.LLM_INVOCATION_PARAMETERS]:
              JSON.stringify(options),
            ...getCompletionInputValueAndMimeType(prompts),
          },
        });
        const execContext = getExecContext(span);

        const execPromise = safeExecuteInTheMiddle(
          () => {
            return context.with(execContext, () => {
              return original.apply(this, args);
            });
          },
          (error) => {
            // Push the error to the span
            if (error) {
              span.recordException(error);
              span.setStatus({
                code: SpanStatusCode.ERROR,
                message: error.message,
              });
              span.end();
            }
          },
        );
        const wrappedPromise = execPromise.then((result) => {
          span.setAttributes({
            ...getCompletionOutputValueAndMimeType(result),
            ...getUsageAttributes(result),
          });
          span.setStatus({ code: SpanStatusCode.OK });
          span.end();
          return result;
        });
        return context.bind(execContext, wrappedPromise);
      };
    });

    // Patch embeddings
    type EmbeddingsCreateType =
      typeof module.OpenAIClient.prototype.getEmbeddings;
    this._wrap(
      module.OpenAIClient.prototype,
      "getEmbeddings",
      (original: EmbeddingsCreateType) => {
        return function patchedEmbeddingCreate(
          this: unknown,
          ...args: Parameters<EmbeddingsCreateType>
        ) {
          const deploymentName = args[0];
          const input = args[1];
          const span = instrumentation.tracer.startSpan(`OpenAI Embeddings`, {
            kind: SpanKind.INTERNAL,
            attributes: {
              [SemanticConventions.OPENINFERENCE_SPAN_KIND]:
                OpenInferenceSpanKind.EMBEDDING,
              [SemanticConventions.EMBEDDING_MODEL_NAME]: deploymentName,
              [SemanticConventions.INPUT_VALUE]: JSON.stringify(input),
              [SemanticConventions.INPUT_MIME_TYPE]: MimeType.JSON,
              ...getEmbeddingTextAttributes(input),
            },
          });
          const execContext = getExecContext(span);
          const execPromise = safeExecuteInTheMiddle<
            ReturnType<EmbeddingsCreateType>
          >(
            () => {
              return context.with(execContext, () => {
                return original.apply(this, args);
              });
            },
            (error) => {
              // Push the error to the span
              if (error) {
                span.recordException(error);
                span.setStatus({
                  code: SpanStatusCode.ERROR,
                  message: error.message,
                });
                span.end();
              }
            },
          );
          const wrappedPromise = execPromise.then((result) => {
            if (result) {
              span.setAttributes({
                ...getEmbeddingVectorAttributes(result),
              });
            }
            span.setStatus({ code: SpanStatusCode.OK });
            span.end();
            return result;
          });
          return context.bind(execContext, wrappedPromise);
        };
      },
    );

    _isOpenInferencePatched = true;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      module.openInferencePatched = true;
    } catch (e) {
      diag.warn(`Failed to set ${MODULE_NAME} patched flag on the module`, e);
    }

    return module;
  }
  /**
   * Un-patches the OpenAI module's chat completions API
   */
  private unpatch(
    moduleExports: typeof AzureOpenAI & { openInferencePatched?: boolean },
    moduleVersion?: string,
  ) {
    diag.debug(`Removing patch for ${MODULE_NAME}@${moduleVersion}`);
    this._unwrap(moduleExports.OpenAI.Chat.Completions.prototype, "create");
    this._unwrap(moduleExports.OpenAI.Completions.prototype, "create");
    this._unwrap(moduleExports.OpenAI.Embeddings.prototype, "create");

    _isOpenInferencePatched = false;
    try {
      // This can fail if the module is made immutable via the runtime or bundler
      moduleExports.openInferencePatched = false;
    } catch (e) {
      diag.warn(`Failed to unset ${MODULE_NAME} patched flag on the module`, e);
    }
  }
}

/**
 * type-guard that checks if the response is a chat completion response
 */
function isChatCompletionResponse(
  response: Stream<ChatCompletions> | ChatCompletions,
): response is ChatCompletions {
  return "choices" in response;
}

/**
 * Consumes the stream chunks and adds them to the span
 */
async function consumeChatCompletionStreamChunks(
  stream: Stream<ChatCompletionChunk>,
  span: Span,
) {
  let streamResponse = "";
  // Tool and function call attributes can also arrive in the stream
  // NB: the tools and function calls arrive in partial diffs
  // So the final tool and function calls need to be aggregated
  // across chunks
  const toolAndFunctionCallAttributes: Attributes = {};
  // The first message is for the assistant response so we start at 1
  for await (const chunk of stream) {
    if (chunk.choices.length <= 0) {
      continue;
    }
    const choice = chunk.choices[0];
    if (choice.delta.content) {
      streamResponse += choice.delta.content;
    }
    // Accumulate the tool and function call attributes
    const toolAndFunctionCallAttributesDiff =
      getToolAndFunctionCallAttributesFromStreamChunk(chunk);
    for (const [key, value] of Object.entries(
      toolAndFunctionCallAttributesDiff,
    )) {
      if (isString(toolAndFunctionCallAttributes[key]) && isString(value)) {
        toolAndFunctionCallAttributes[key] += value;
      } else if (isString(value)) {
        toolAndFunctionCallAttributes[key] = value;
      }
    }
  }
  const messageIndexPrefix = `${SemanticConventions.LLM_OUTPUT_MESSAGES}.0.`;

  // Append the attributes to the span as a message
  const attributes: Attributes = {
    [SemanticConventions.OUTPUT_VALUE]: streamResponse,
    [SemanticConventions.OUTPUT_MIME_TYPE]: MimeType.TEXT,
    [`${messageIndexPrefix}${SemanticConventions.MESSAGE_CONTENT}`]:
      streamResponse,
    [`${messageIndexPrefix}${SemanticConventions.MESSAGE_ROLE}`]: "assistant",
  };
  // Add the tool and function call attributes
  for (const [key, value] of Object.entries(toolAndFunctionCallAttributes)) {
    attributes[`${messageIndexPrefix}${key}`] = value;
  }
  span.setAttributes(attributes);
  span.end();
}

/**
 * Extracts the semantic attributes from the stream chunk for tool_calls and function_calls
 */
function getToolAndFunctionCallAttributesFromStreamChunk(
  chunk: ChatCompletionChunk,
): Attributes {
  if (chunk.choices.length <= 0) {
    return {};
  }
  const choice = chunk.choices[0];
  const attributes: Attributes = {};
  if (choice.delta.tool_calls) {
    choice.delta.tool_calls.forEach((toolCall, index) => {
      const toolCallIndexPrefix = `${SemanticConventions.MESSAGE_TOOL_CALLS}.${index}.`;
      // Double check that the tool call has a function
      // NB: OpenAI only supports tool calls with functions right now but this may change
      if (toolCall.function) {
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
  if (choice.delta.function_call) {
    attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_NAME] =
      choice.delta.function_call.name;
    attributes[SemanticConventions.MESSAGE_FUNCTION_CALL_ARGUMENTS_JSON] =
      choice.delta.function_call.arguments;
  }
  return attributes;
}
