// File: backend/agentos/core/llm/providers/implementations/OpenAIProvider.ts

/**
 * @fileoverview Implements the IProvider interface for OpenAI's GPT models.
 * This provider offers comprehensive integration with OpenAI's API, including:
 * - Chat completions (streaming and non-streaming)
 * - Tool/function calling
 * - Vision capabilities for multimodal models (e.g., GPT-4o)
 * - Text embeddings generation
 * - Model introspection and health checks.
 *
 * It is designed with robustness and extensibility in mind, featuring:
 * - Detailed error handling using custom `OpenAIProviderError`.
 * - API key management with support for system-wide and per-request overrides.
 * - Configurable request retries with exponential backoff.
 * - Rate limiting (conceptual, actual enforcement by OpenAI).
 * - Comprehensive JSDoc documentation.
 * - Adherence to TypeScript best practices and modern ECMAScript features.
 *
 * @module backend/agentos/core/llm/providers/implementations/OpenAIProvider
 * @implements {IProvider}
 */

import {
  IProvider,
  ChatMessage,
  ModelCompletionOptions,
  ModelCompletionResponse,
  ModelCompletionChoice,
  ModelInfo,
  ModelUsage,
  ProviderEmbeddingOptions,
  ProviderEmbeddingResponse,
} from '../IProvider';
import { stripForeignVendorParams } from '../openrouter-only-params';
import { resolveOpenAiCacheRetentionParams, resolvePromptCacheKey } from '../openai-cache-params';
import { OpenAIProviderError } from '../errors/OpenAIProviderError';
import { ApiKeyPool } from '../../../providers/ApiKeyPool.js';
import { toOpenAiResponseFormat } from './openai-response-format-guard';
import { clampMaxOutputTokens } from '../model-output-limits.js';
import { mapEffortToOpenAiReasoningEffort, mapEffortToOpenAiReasoningEffortForModel, mapEffortToOpenAiResponsesEffort } from '../model-effort.js';
import { computeRetryBackoffMs } from './retry-backoff.js';
import { redactUrlSecrets } from '../url-secrets.js';
// Assuming a fetch-like interface is available globally or polyfilled (e.g., node-fetch)
// For Node.js, ensure 'node-fetch' is a dependency or use Node's built-in fetch from v18+.
// import fetch, { RequestInit, Response as FetchResponse, AbortController } from 'node-fetch'; // Example for Node

/**
 * Minimal structural types for the subset of OpenAI REST responses this provider
 * consumes, defined from the fields the mapping code accesses (the API returns a
 * superset). Previously the whole file was `@ts-nocheck`, which silently masked
 * this missing namespace and the unguarded `choices` access in the mappers.
 */
// eslint-disable-next-line @typescript-eslint/no-namespace
namespace OpenAIAPITypes {
  export interface Usage {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
    /**
     * OpenAI automatic prompt caching (gpt-4o and newer): prompt tokens
     * served from cache at the discounted rate, plus — on GPT-5.6+ —
     * tokens written to cache (billed 1.25x). Normalized into
     * {@link ModelUsage.cacheReadInputTokens} / cacheCreationInputTokens
     * so caching is visible platform-wide (OpenRouterProvider already
     * normalizes the same fields).
     */
    prompt_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number };
  }
  /** Complete tool call as returned on a non-streaming message. */
  export interface ToolCall {
    id: string;
    type: 'function';
    function: { name: string; arguments: string };
  }
  /** Partial tool call as returned on a streaming delta. */
  export interface ToolCallDelta {
    index: number;
    id?: string;
    type?: 'function';
    function?: { name?: string; arguments?: string };
  }
  export interface ChatMessage {
    role: string;
    content: string | null;
    tool_calls?: ToolCall[];
  }
  export interface ChatChoice {
    index: number;
    message: ChatMessage;
    finish_reason: string | null;
    logprobs?: unknown;
  }
  export interface ChatCompletionResponse {
    id: string;
    object: string;
    created: number;
    model: string;
    choices: ChatChoice[];
    usage?: Usage;
    /** Service tier the call actually ran at (present when service_tier was requested or defaulted). */
    service_tier?: string;
  }
  export interface StreamChunkExtras {
    /** Service tier the call actually ran at (present on chunks when requested/defaulted). */
    service_tier?: string;
  }
  export interface StreamDelta {
    role?: string;
    content?: string | null;
    tool_calls?: ToolCallDelta[];
  }
  export interface StreamChoice {
    index: number;
    delta: StreamDelta;
    finish_reason: string | null;
    logprobs?: unknown;
  }
  export interface ChatCompletionStreamResponse extends StreamChunkExtras {
    id: string;
    object: string;
    created: number;
    model: string;
    // Optional: the trailing usage-only chunk (stream_options.include_usage)
    // carries an empty/absent choices array.
    choices?: StreamChoice[];
    usage?: Usage;
  }
  export interface ModelAPIObject {
    id: string;
    object?: string;
    created?: number;
    owned_by?: string;
  }
  export interface ListModelsResponse {
    object?: string;
    data: ModelAPIObject[];
  }
  export interface EmbeddingItem {
    object?: string;
    index: number;
    embedding: number[];
  }
  export interface EmbeddingResponse {
    object?: string;
    model: string;
    data: EmbeddingItem[];
    usage: Usage;
  }
  export interface APIErrorResponse {
    error?: { message?: string; type?: string; code?: string };
  }

  // --- /v1/responses (Responses API) — narrow read surface --------------
  // Only the fields mapResponsesToCompletionResponse reads. Live-probed
  // 2026-07-08 (gpt-5.5): output items carry `type`, message content is
  // `output_text` parts, function_call items carry `call_id`/`name`/
  // `arguments`, usage uses input_tokens/output_tokens, `created_at` +
  // `status` + `incomplete_details` are top-level.
  export interface ResponsesUsage {
    input_tokens: number;
    output_tokens: number;
    total_tokens: number;
    /** Responses-API spelling of the cached-prompt-token detail. */
    input_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number };
  }
  export interface ResponsesOutputContentPart {
    type: string; // 'output_text' | …
    text?: string;
  }
  export interface ResponsesOutputItem {
    type: string; // 'message' | 'function_call' | 'reasoning' | …
    // message item
    role?: string;
    content?: ResponsesOutputContentPart[];
    // function_call item
    call_id?: string;
    name?: string;
    arguments?: string;
  }
  export interface ResponsesResponse {
    id: string;
    object?: string; // 'response'
    created_at?: number;
    model?: string;
    status?: string; // 'completed' | 'incomplete' | …
    output?: ResponsesOutputItem[];
    usage?: ResponsesUsage;
    incomplete_details?: { reason?: string } | null;
    /** Service tier the call actually ran at. */
    service_tier?: string;
  }
}

/**
 * Configuration specific to the OpenAIProvider.
 */
export interface OpenAIProviderConfig {
  /** The API key for accessing OpenAI services. Can be overridden by `apiKeyOverride` in request options. Can be empty when oauthFlow is provided. */
  apiKey: string;
  /** Base URL for the OpenAI API. Defaults to "https://api.openai.com/v1". Useful for proxies. */
  baseURL?: string;
  /** Default OpenAI organization ID to use for requests. */
  organizationId?: string;
  /** Maximum number of retry attempts for failed API requests. Defaults to 3. */
  maxRetries?: number;
  /** Timeout for API requests in milliseconds. Defaults to 60000 (60 seconds). */
  requestTimeout?: number;
  /** Optional custom headers to include with all requests to the OpenAI API. */
  customHeaders?: Record<string, string>;
  /** Default model ID to use if not specified in a request. E.g., "gpt-4o-mini". */
  defaultModelId?: string;
  /** Optional OAuth flow for subscription-based auth (e.g. ChatGPT Plus/Pro). When set, apiKey is derived dynamically from OAuth tokens. */
  oauthFlow?: { getAccessToken(): Promise<string> };
}

// Simplified OpenAI API response structures (non-namespaced)
type OpenAIChatCompletionChoice = {
    index: number;
    message: {
      role: ChatMessage['role'];
      content: string | null;
      tool_calls?: Array<{
        id: string;
        type: 'function';
      function: { name: string; arguments: string };
      }>;
    };
    finish_reason: string | null;
    logprobs?: unknown;
};

type _OpenAIChatCompletionResponse = {
    id: string;
    object: string; // e.g., "chat.completion"
    created: number; // Unix timestamp
    model: string; // Model ID used
  choices: OpenAIChatCompletionChoice[];
    usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
    system_fingerprint?: string;
};

type OpenAIChatCompletionStreamChoiceDelta = {
    role?: ChatMessage['role'];
    content?: string;
    tool_calls?: Array<{
      index: number; // Required for accumulating tool calls
      id?: string;
      type?: 'function';
    function?: { name?: string; arguments?: string };
    }>;
};

type OpenAIChatCompletionStreamChoice = {
    index: number;
  delta: OpenAIChatCompletionStreamChoiceDelta;
    finish_reason?: string | null;
    logprobs?: unknown;
};

type _OpenAIChatCompletionStreamResponse = {
    id: string;
    object: string; // e.g., "chat.completion.chunk"
    created: number;
    model: string;
  choices: OpenAIChatCompletionStreamChoice[];
  usage?: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
    };
    system_fingerprint?: string;
};

type OpenAIEmbeddingAPIObject = {
    object: 'embedding';
    embedding: number[];
    index: number;
};

type _OpenAIEmbeddingResponse = {
    object: 'list';
  data: OpenAIEmbeddingAPIObject[];
    model: string; // Model ID used
    usage: {
      prompt_tokens: number;
      total_tokens: number;
    };
};

type OpenAIModelAPIObject = {
    id: string;
    object: 'model';
    created: number; // Unix timestamp
    owned_by: string;
};

type _OpenAIListModelsResponse = {
    object: 'list';
  data: OpenAIModelAPIObject[];
};

type _OpenAIAPIErrorResponse = {
    error?: {
      message: string;
      type: string;
      param?: string | null;
      code?: string | null;
    };
};

/**
 * @class OpenAIProvider
 * @implements {IProvider}
 * Provides an interface to OpenAI's suite of models (GPT, Embeddings).
 * It handles API requests, streaming, error management, and model information.
 */
/**
 * Whether the given model id belongs to the family that requires
 * `max_completion_tokens` instead of the legacy `max_tokens` parameter.
 *
 * OpenAI's reasoning models (o1, o3, o4) and the GPT-5 family reject
 * `max_tokens` outright with HTTP 400 "Unsupported parameter:
 * 'max_tokens' is not supported with this model. Use
 * 'max_completion_tokens' instead." Legacy chat completions
 * (gpt-4o, gpt-4-turbo, gpt-4.1, gpt-3.5, etc.) still accept the
 * old field.
 *
 * Errs on the conservative side — any model id that is not a clear
 * member of one of the new families uses `max_tokens` so older
 * deployments do not silently break when the param-name flag changes.
 *
 * @param modelId Provider-side model identifier (e.g. `'gpt-5.4-mini'`).
 * @returns `true` when the model needs `max_completion_tokens`.
 */
export function modelRequiresMaxCompletionTokens(modelId: string): boolean {
  return isOpenAIReasoningModel(modelId);
}

/**
 * Context-window size for the GPT-5, GPT-6 and o-series reasoning families.
 *
 * From the first-party model pages and GET /v1/models, 2026-09-23 to 29. The
 * family prefix alone does not decide the size: `gpt-5.4` is a 1.05M model
 * while its `-mini` and `-nano` siblings are 400K.
 *
 *   - 1,050,000: the GPT-6 family (`gpt-6.1-sol` included, per the
 *     2026-09-29 changelog), the GPT-5.6 family except `gpt-5.6-cyber`,
 *     `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4` and `gpt-5.4-pro` (max input
 *     922,000 on GPT-6).
 *   - 400,000: `gpt-5.6-cyber`, `gpt-5.4-mini` / `-nano`, `gpt-5.3-codex`,
 *     `gpt-5.2*`, `gpt-5.1*`, `gpt-5`, and `gpt-5-mini` / `-nano` / `-pro`.
 *   - 200,000: the o-series other than `o1-mini` and `o1-preview`.
 *   - 128,000: `o1-mini`, `o1-preview` and the `gpt-5*-chat-latest` snapshots.
 *
 * @param modelId Provider-side model identifier (e.g. `'gpt-6-sol'`).
 * @returns Context window in tokens.
 */
export function openAiReasoningContextWindow(modelId: string): number {
  const id = modelId.toLowerCase();
  if (/^o1-(mini|preview)/.test(id)) return 128000;
  if (/^o\d/.test(id)) return 200000;
  if (/^gpt-5(\.\d+)?-chat-latest$/.test(id)) return 128000;
  if (/^gpt-5\.6-cyber/.test(id)) return 400000;
  if (/^gpt-6/.test(id)) return 1050000;
  if (/^gpt-5\.[56]/.test(id)) return 1050000;
  // gpt-5.4 and gpt-5.4-pro, bare or dated, are 1.05M; -mini and -nano are 400K.
  if (/^gpt-5\.4(-pro)?(-\d{4}-\d{2}-\d{2})?$/.test(id)) return 1050000;
  return 400000;
}

/**
 * Prompt size, in input tokens, above which OpenAI bills a call on a
 * long-context model at the long-context rates. The pricing page splits its
 * flagship table into "<=272K" and ">272K" input tokens, and the comparison is
 * strict: a prompt of exactly 272,000 tokens bills at the standard rates. The
 * count includes cached tokens, and output size never triggers the tier.
 */
export const OPENAI_LONG_CONTEXT_THRESHOLD_TOKENS = 272_000;

/**
 * Factor OpenAI applies to the input rate of a long-context call. The cached
 * input and cache write rates double as well.
 */
export const OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER = 2;

/** Factor OpenAI applies to the output rate of a long-context call. */
export const OPENAI_LONG_CONTEXT_OUTPUT_MULTIPLIER = 1.5;

/**
 * Whether OpenAI bills `modelId` at the long-context rates when a prompt
 * exceeds {@link OPENAI_LONG_CONTEXT_THRESHOLD_TOKENS} input tokens. Such a
 * call bills its whole input at {@link OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER}
 * times the input rate and its whole output at
 * {@link OPENAI_LONG_CONTEXT_OUTPUT_MULTIPLIER} times the output rate.
 *
 * OpenAI states the rule for its models with a 1.05M context window, so this
 * reuses {@link openAiReasoningContextWindow}. That covers the GPT-6 family
 * (`gpt-6.1-sol` included), the GPT-5.6 family except `gpt-5.6-cyber`,
 * `gpt-5.5`, `gpt-5.5-pro`, `gpt-5.4` and `gpt-5.4-pro`, bare or dated.
 * Everything else bills flat at every prompt size: the GPT-5 models with a
 * smaller window (`gpt-5.6-cyber`, which the pricing page lists without a
 * long-context rate, `gpt-5.4-mini` / `-nano`, `gpt-5.3` and older, and the
 * chat-latest snapshots), the o-series, and the pre-GPT-5 models, `gpt-4.1`
 * included despite its 1M window. Checked against
 * developers.openai.com/api/docs/pricing and the model pages on 2026-09-30.
 *
 * @param modelId Model id as requested or as echoed by the API.
 * @returns `true` when a prompt above the threshold bills at the long-context rates.
 */
export function openAiHasLongContextPricing(modelId: string): boolean {
  return isOpenAIReasoningModel(modelId) && openAiReasoningContextWindow(modelId) === 1050000;
}

/**
 * Whether OpenAI serves `modelId` only through the Responses API. OpenAI lists
 * the `-pro` tiers, the codex models, `gpt-5.6-cyber` and the deep-research
 * models as Responses-only. OpenAIProvider sends a request to /v1/responses
 * only when it is a non-streamed GPT-5 or GPT-6 call carrying function tools
 * and an effort, and to /v1/chat/completions otherwise, where these ids fail.
 *
 * @param modelId Provider-side model identifier.
 * @returns `true` when the model has no Chat Completions endpoint.
 */
export function isOpenAIResponsesOnlyModel(modelId: string): boolean {
  return /-pro(-\d{4}-\d{2}-\d{2})?$|codex|^gpt-5\.6-cyber|deep-research/i.test(modelId);
}

/**
 * ModelInfo capabilities for a GPT-5, GPT-6 or o-series id, limited to what
 * OpenAIProvider can serve.
 *
 *   - Responses-only models ({@link isOpenAIResponsesOnlyModel}) list as
 *     `chat` only. This provider reaches them only on its narrow Responses
 *     path, and the refresh leaves the Responses-only o-series ids out.
 *   - GPT-6 omits `tool_use`. OpenAI requires the Responses API for GPT-6
 *     Astra tool calls and `reasoning_effort: "none"` for Sol and Luna tool
 *     calls on Chat Completions, and this provider routes a tool call to
 *     Responses only when it carries an effort and is not streamed.
 *   - `o1-mini` and `o1-preview` have no function calling or structured
 *     outputs. They and `o3-mini` take text only, and every other id takes
 *     images (per model input modalities, checked 2026-09-29).
 *
 * @param modelId Provider-side model identifier (e.g. `'gpt-6-sol'`).
 * @returns Capability strings for ModelInfo.
 */
export function openAiReasoningCapabilities(modelId: string): string[] {
  const id = modelId.toLowerCase();
  if (isOpenAIResponsesOnlyModel(id) || /^o1-(mini|preview)/.test(id)) return ['chat'];
  const caps = ['chat', 'json_mode'];
  if (!/^gpt-6/.test(id)) caps.push('tool_use');
  if (!/^o3-mini/.test(id)) caps.push('vision_input');
  return caps;
}

/**
 * Whether `modelId` is an OpenAI reasoning model — the o1/o3/o4 series or the
 * GPT-5 / GPT-6 families. These models reject custom sampling params
 * (`temperature`, `top_p`) with HTTP 400 in addition to requiring
 * `max_completion_tokens`; they run only at their fixed defaults. Single
 * source of truth for the o-series/gpt-5/gpt-6 detection used by both the
 * param guards.
 *
 * GPT-6 live-probed 2026-09-10 (chat.completions, `gpt-6-astra`):
 * `temperature: 0.5` → HTTP 400 `unsupported_value` ("Unsupported value:
 * 'temperature' does not support 0.5 with this model. Only the default (1)
 * value is supported."); `max_tokens: 16` → HTTP 400
 * `unsupported_parameter` ("Unsupported parameter: 'max_tokens' is not
 * supported with this model. Use 'max_completion_tokens' instead."). The
 * family therefore behaves exactly like GPT-5 on both guards.
 *
 * `gpt-6` is matched as a bare family prefix (mirroring `gpt-5`) so dated
 * snapshots of `gpt-6-astra` are covered. It is deliberately NOT generalized
 * to `gpt-\d` — a future family's param contract is unknown, and the
 * conservative default for an unrecognized id is the legacy `max_tokens`
 * path. Widen only for a family that has actually shipped and been probed.
 *
 * @param modelId Provider-side model identifier (e.g. `'o3'`, `'gpt-6-astra'`).
 */
export function isOpenAIReasoningModel(modelId: string): boolean {
  // o1 / o3 / o4 reasoning models, plus the GPT-5 and GPT-6 families.
  return /^(o\d|gpt-5|gpt-6)/i.test(modelId);
}

/**
 * Whether OpenAI rejects `reasoning_effort` sent ALONGSIDE function tools on
 * the `/v1/chat/completions` endpoint for `modelId`. The GPT-5 family enforces
 * this — "Function tools with reasoning_effort are not supported for gpt-5.5
 * in /v1/chat/completions. Please use /v1/responses instead." (live-observed
 * 2026-07-08 when the codegen frontier fallback chain routed a tool-carrying,
 * effort:xhigh orchestrator turn to gpt-5.5, hard-erroring the whole build).
 *
 * Such a call is routed to `/v1/responses` when it qualifies (see
 * {@link shouldRouteToOpenAiResponsesApi}, which preserves BOTH effort and
 * tools); otherwise it stays on chat/completions and must DROP
 * `reasoning_effort` (the model then reasons at its default depth) rather than
 * 400 the entire request. Scoped to the GPT-5 and GPT-6 families: the o-series
 * (o1/o3/o4) still accepts `reasoning_effort` + tools on chat/completions, so
 * dropping it there would be a needless reasoning-depth regression. Widen this
 * predicate if a future o-series enforces the same Responses-API requirement.
 *
 * GPT-6 enforces the SAME restriction — live-probed 2026-09-10
 * (chat.completions, `gpt-6-astra`, `reasoning_effort: 'xhigh'` + one function
 * tool): HTTP 400 `invalid_request_error` — "Function tools with
 * reasoning_effort are not supported for gpt-6-astra in /v1/chat/completions.
 * To use function tools, use /v1/responses or set reasoning_effort to 'none'."
 * The suggested `'none'` escape hatch does NOT exist on this family: the same
 * probe with `reasoning_effort: 'none'` returns HTTP 400 `unsupported_value`
 * ("Supported values are: 'low', 'medium', 'high', and 'xhigh'"). Dropping the
 * param entirely (or routing to `/v1/responses`) is therefore the only viable
 * path, which is exactly what the two call sites below already do.
 *
 * @param modelId Provider-side model identifier (e.g. `'gpt-5.5'`, `'gpt-6-astra'`).
 */
export function openAiRejectsReasoningEffortWithTools(modelId: string): boolean {
  return /^gpt-(5|6)/i.test(modelId);
}

/**
 * Whether the request carries function tools (present + non-empty). Used to
 * decide whether the {@link openAiRejectsReasoningEffortWithTools} guard
 * applies — reasoning_effort is only incompatible WITH tools.
 */
function requestHasFunctionTools(tools: unknown): boolean {
  return Array.isArray(tools) ? tools.length > 0 : tools != null;
}

/** Whether one content part is a plain text block the Responses mapper can carry. */
function isTextPart(part: unknown): part is { type: 'text'; text: string } {
  return (
    part != null &&
    typeof part === 'object' &&
    (part as { type?: unknown }).type === 'text' &&
    typeof (part as { text?: unknown }).text === 'string'
  );
}

/**
 * Whether a message's `content` can be carried to `/v1/responses` by
 * {@link flattenResponsesTextContent} WITHOUT losing information.
 *
 * Accepts a plain string, `null`/absent content, or an array whose parts are
 * ALL text blocks. Anthropic-style `cache_control` markers on those blocks are
 * fine to drop: OpenAI has no equivalent request field and caches long prefixes
 * automatically, so the marker carries no payload.
 *
 * Rejects genuinely multimodal content (images, `tool_result` blocks, any
 * non-text part) because the mapper still cannot represent it — flattening
 * those would silently discard the payload.
 *
 * This is the SINGLE source of truth shared by
 * {@link shouldRouteToOpenAiResponsesApi} (which refuses to route what the
 * mapper cannot carry) and the mapper itself, so the two cannot drift into the
 * state that caused the silent-empty-system-prompt hazard this guard replaced.
 */
export function isResponsesMappableContent(content: ChatMessage['content']): boolean {
  if (content == null || typeof content === 'string') return true;
  if (!Array.isArray(content)) return false;
  return content.every(isTextPart);
}

/**
 * Flatten a message's content to the plain text `/v1/responses` receives.
 *
 * A string passes through; an all-text block array is joined with `\n` (the
 * shape `systemBlocks` / `SystemContentBlock[]` produces, where each block is
 * an independently cache-markable slab of one logical prompt); anything else
 * yields `''`.
 *
 * Callers MUST gate on {@link isResponsesMappableContent} first — the `''`
 * fallback is a defensive floor for unmappable content, not a licence to send
 * it. Sending unmappable content here is exactly the silent-empty-prompt bug
 * the paired predicate exists to prevent.
 */
export function flattenResponsesTextContent(content: ChatMessage['content']): string {
  if (typeof content === 'string') return content;
  if (content == null || !Array.isArray(content)) return '';
  return content.filter(isTextPart).map((p) => p.text).join('\n');
}

/**
 * Whether a NON-streaming call must use `/v1/responses` instead of
 * `/v1/chat/completions`: a gpt-5/gpt-6 reasoning model carrying function tools
 * AND a requested effort — the ONLY combination chat/completions 400s on
 * ("Function tools with reasoning_effort are not supported … Please use
 * /v1/responses instead") and the only case where the Responses path buys
 * anything (preserved reasoning depth). Excludes, per the 2026-07-08 Codex
 * review, requests that carry a `responseFormat` (the Responses path maps no
 * structured output — those stay on chat/completions where R4 drops effort) or
 * message content the input mapper cannot carry (see
 * {@link isResponsesMappableContent}). Every excluded case stays on
 * chat/completions; R4's effort-drop remains the defense-in-depth floor there.
 *
 * Content gating is delegated to {@link isResponsesMappableContent} rather
 * than tested inline. It previously required EVERY message's content to be a
 * plain string, which silently excluded Anthropic-style cache-marked system
 * blocks (`systemBlocks` / `SystemContentBlock[]`, which reach the provider as
 * an all-text content array). Callers that set `systemBlocks` to fix a cold
 * prompt cache therefore lost Responses routing and hard-400ed on
 * chat/completions — the exact failure this router exists to prevent. All-text
 * arrays now route and are flattened losslessly; true multimodal content is
 * still excluded, because the mapper still cannot represent it.
 */
export function shouldRouteToOpenAiResponsesApi(
  modelId: string,
  messages: ChatMessage[],
  options: Pick<ModelCompletionOptions, 'tools' | 'effort' | 'responseFormat'>,
): boolean {
  return (
    openAiRejectsReasoningEffortWithTools(modelId) &&
    requestHasFunctionTools(options.tools) &&
    mapEffortToOpenAiReasoningEffort(options.effort) !== undefined &&
    options.responseFormat === undefined &&
    messages.every((m) => isResponsesMappableContent(m.content))
  );
}

export class OpenAIProvider implements IProvider {
  /** @inheritdoc */
  public readonly providerId: string = 'openai';
  /** @inheritdoc */
  public isInitialized: boolean = false;
  /** @inheritdoc */
  public defaultModelId?: string;

  private config!: OpenAIProviderConfig; // Asserted as initialized by `initialize`
  private keyPool: ApiKeyPool | null = null;
  private availableModelsCache: Map<string, ModelInfo> = new Map();

  // Known pricing for common OpenAI models (USD per 1K tokens).
  // Verified against openai.com/api/pricing and developers.openai.com/api/docs/pricing
  // on 2026-04-16. Values are standard (non-batch, non-regional) rates.
  // Input: cost for prompt tokens. Output: cost for completion tokens.
  // For embedding models, 'input' is total tokens.
  // A prompt of more than 272,000 input tokens on a 1.05M-context model bills
  // the whole call at 2x input and 1.5x output
  // (developers.openai.com/api/docs/pricing, 2026-09-30). Rows hold the rates
  // below that threshold, and calculateCost applies the tier from
  // openAiHasLongContextPricing.
  private readonly modelPricing: Record<string, { input: number; output: number }> = {
    // GPT-6 family (current flagship, Sep 2026). Astra $10/$50, Sol $2/$10,
    // Luna $0.10/$0.50 per 1M, from developers.openai.com/api/docs/models on
    // 2026-09-23; all three ids are on the first-party GET /v1/models listing.
    // A model with no row here gets costUSD undefined from calculateCost, so
    // its calls go unmetered. "GPT-6 Pro" is a ChatGPT plan tier, and the
    // first-party pro tier is `reasoning.mode: 'pro'` on Sol, so neither has an
    // id here.
    'gpt-6-astra': { input: 0.01, output: 0.05 },
    'gpt-6-sol': { input: 0.002, output: 0.01 },
    'gpt-6-luna': { input: 0.0001, output: 0.0005 },
    // gpt-6.1-sol (developers.openai.com/api/docs/changelog, 2026-09-29): $2
    // input and $10 output per 1M like gpt-6-sol, with cached input at $0.10
    // (5% of input, where the other GPT-6 models charge 10%) and cache writes
    // at $2.50. This table carries no cached or cache-write column.
    'gpt-6.1-sol': { input: 0.002, output: 0.01 },
    // GPT-5.5 family (previous flagship, Jun 2026 — $5 / $30 per 1M tokens, a 2x
    // increase over gpt-5.4; verified against OpenAI's published pricing 2026-06-27)
    'gpt-5.5': { input: 0.005, output: 0.03 },
    'gpt-5.5-pro': { input: 0.03, output: 0.18 },
    // GPT-5.6 family (Aug 2026), per the first-party model pages on
    // 2026-09-23: Sol $4/$20 (promotional through at least 2026-11-21),
    // Terra $2/$12 and Luna $0.20/$1.20 per 1M. OpenRouter's listing carries
    // OpenRouter's resale rates, which differ, so first-party providers are
    // priced from first-party pages. The bare `gpt-5.6` alias routes to Sol and
    // is priced with it. It is absent from the 2026-09-23 /v1/models listing,
    // but aliases are often unlisted and this repo records a direct HTTP 200 on
    // it from 2026-08-06 (see RESPONSES_MAX_EFFORT_MODELS), so it stays until an
    // inference probe shows it gone.
    'gpt-5.6': { input: 0.004, output: 0.02 },
    'gpt-5.6-sol': { input: 0.004, output: 0.02 },
    'gpt-5.6-terra': { input: 0.002, output: 0.012 },
    'gpt-5.6-luna': { input: 0.0002, output: 0.0012 },
    // GPT-5.4 family (previous flagship and siblings, Mar 2026)
    'gpt-5.4': { input: 0.0025, output: 0.015 },
    'gpt-5.4-mini': { input: 0.00075, output: 0.0045 },
    'gpt-5.4-nano': { input: 0.0002, output: 0.00125 },
    'gpt-5.4-pro': { input: 0.030, output: 0.180 },
    // GPT-5.3 family (ChatGPT / Codex variants, priced per Azure/OpenAI listings)
    'gpt-5.3-chat-latest': { input: 0.00175, output: 0.014 },
    'gpt-5.3-codex': { input: 0.00175, output: 0.014 },
    // GPT-5.2 family
    'gpt-5.2': { input: 0.00175, output: 0.014 },
    // GPT-5.1 and GPT-5 base (legacy, same tier pricing)
    'gpt-5': { input: 0.00125, output: 0.010 },
    'gpt-5.1': { input: 0.00125, output: 0.010 },
    'gpt-5-mini': { input: 0.00025, output: 0.002 },
    'gpt-5-nano': { input: 0.00005, output: 0.0004 },
    // GPT-4.1 family (legacy but still in rotation)
    'gpt-4.1': { input: 0.002, output: 0.008 },
    'gpt-4.1-mini': { input: 0.0004, output: 0.0016 },
    'gpt-4.1-nano': { input: 0.0001, output: 0.0004 },
    // GPT-4o family (legacy, repriced to current list at $2.50/$10 per 1M)
    'gpt-4o': { input: 0.0025, output: 0.010 },
    // The first GPT-4o snapshot kept its $5 / $15 launch price. Without this
    // row the dated-snapshot fallback would price it at the gpt-4o alias rate.
    'gpt-4o-2024-05-13': { input: 0.005, output: 0.015 },
    'gpt-4o-mini': { input: 0.00015, output: 0.0006 },
    // Deprecated but still potentially referenced
    'gpt-4-turbo': { input: 0.01, output: 0.03 },
    'gpt-4-turbo-preview': { input: 0.01, output: 0.03 },
    'gpt-4-vision-preview': { input: 0.01, output: 0.03 },
    'gpt-4': { input: 0.03, output: 0.06 },
    'gpt-4-32k': { input: 0.06, output: 0.12 },
    'gpt-3.5-turbo': { input: 0.0005, output: 0.0015 },
    'gpt-3.5-turbo-16k': { input: 0.001, output: 0.002 },
    // o-series reasoning models
    'o3': { input: 0.002, output: 0.008 },
    'o3-mini': { input: 0.0011, output: 0.0044 },
    'o3-pro': { input: 0.020, output: 0.080 },
    'o3-pro-2025-06-10': { input: 0.020, output: 0.080 },
    'o4-mini': { input: 0.0011, output: 0.0044 },
    'o1': { input: 0.015, output: 0.060 },
    'o1-pro': { input: 0.150, output: 0.600 },
    // Embedding models (per 1K tokens, input only)
    'text-embedding-3-large': { input: 0.00013, output: 0 },
    'text-embedding-3-small': { input: 0.00002, output: 0 },
    'text-embedding-ada-002': { input: 0.0001, output: 0 },
  };


  /**
   * Creates an instance of OpenAIProvider.
   * Note: The provider is not ready to use until `initialize()` is called and resolves.
   */
  constructor() {
    // Configuration is set during initialize()
  }

  /** @inheritdoc */
  public async initialize(config: OpenAIProviderConfig): Promise<void> {
    if (!config.apiKey && !config.oauthFlow) {
      throw new OpenAIProviderError(
        'API key or OAuth flow is required for OpenAIProvider initialization.',
        'INIT_FAILED_MISSING_API_KEY'
      );
    }

    this.config = {
      baseURL: 'https://api.openai.com/v1',
      maxRetries: 3,
      requestTimeout: 60000, // 60 seconds
      ...config, // User-provided config overrides defaults
    };
    this.keyPool = new ApiKeyPool(config.apiKey);
    this.defaultModelId = config.defaultModelId;

    try {
      // Attempt to list models to verify API key and connectivity.
      await this.refreshAvailableModels();
      this.isInitialized = true;
      const env = typeof process !== 'undefined' ? process.env : undefined;
      const debugOn = !!env && (env.AGENTOS_DEBUG === '1' || env.AGENTOS_DEBUG === 'true' || (env.AGENTOS_LOG_LEVEL ?? '').toLowerCase() === 'debug');
      if (debugOn) {
        console.log(`OpenAIProvider initialized successfully. Default model: ${this.defaultModelId || 'Not set'}. Found ${this.availableModelsCache.size} models.`);
      }
    } catch (error: unknown) {
      this.isInitialized = false;
      if (error instanceof OpenAIProviderError) {
        throw error; // Re-throw if already an OpenAIProviderError
      }
      const message = error instanceof Error ? error.message : 'Unknown error during initialization.';
      throw new OpenAIProviderError(
        `OpenAIProvider initialization failed: ${message}`,
        'INITIALIZATION_FAILED',
        undefined, undefined, undefined, error
      );
    }
  }

  /**
   * Fetches the list of available models from OpenAI and updates the internal cache.
   * @private
   * @throws {OpenAIProviderError} If fetching or parsing models fails.
   */
  private async refreshAvailableModels(): Promise<void> {
    const apiKey = await this.getApiKey();
    const response = await this.makeApiRequest<OpenAIAPITypes.ListModelsResponse>(
      '/models',
      'GET',
      apiKey
    );

    this.availableModelsCache.clear();
    response.data.forEach((apiModel: OpenAIAPITypes.ModelAPIObject) => {
      // Basic filtering for generally usable models, can be expanded.
      // The o-series ids neither start with `gpt-` nor contain `embedding`,
      // so they are admitted on their own terms: priced in `modelPricing`
      // (dated snapshots resolve to their base row) and reachable on Chat
      // Completions. That admits `o1`, `o3`, `o3-mini` and `o4-mini`, and
      // leaves out the Responses-only `o1-pro` / `o3-pro` and the unpriced
      // `o1-mini` / `o1-preview`, which would fail when called or list as free.
      const admitOSeries = /^o\d/.test(apiModel.id)
        && !isOpenAIResponsesOnlyModel(apiModel.id)
        && this.resolvePricing(apiModel.id) !== undefined;
      if (apiModel.id.startsWith('gpt-') || admitOSeries || apiModel.id.includes('embedding')) {
        const modelInfo = this.mapApiToModelInfo(apiModel);
        this.availableModelsCache.set(modelInfo.modelId, modelInfo);
      }
    });
  }

  /**
   * Maps an OpenAI API model object to the standard ModelInfo interface.
   * @private
   * @param {OpenAIAPITypes.ModelAPIObject} apiModel - The model object from OpenAI API.
   * @returns {ModelInfo} The standardized ModelInfo object.
   */
  private mapApiToModelInfo(apiModel: OpenAIAPITypes.ModelAPIObject): ModelInfo {
    const capabilities: ModelInfo['capabilities'] = [];
    let contextWindowSize: number | undefined;
    let _supportsTools = false;
    let _supportsVision = false;
    let isEmbeddingModel = false;

    // Infer capabilities and context window from model ID (common OpenAI patterns).
    //
    // The GPT-5, GPT-6 and o-series branch comes first. Without it these
    // families fall through to the trailing `else`, which publishes them as
    // `capabilities: ['chat']` with no context window, and
    // `listAvailableModels({ capability: 'tool_use' })` then omits every
    // current OpenAI model while still returning gpt-4o.
    if (/^(gpt-5|gpt-6|o\d)/i.test(apiModel.id)) {
        capabilities.push(...openAiReasoningCapabilities(apiModel.id));
        contextWindowSize = openAiReasoningContextWindow(apiModel.id);
        _supportsTools = capabilities.includes('tool_use');
        _supportsVision = capabilities.includes('vision_input');
    } else if (apiModel.id.startsWith('gpt-4o')) {
        capabilities.push('chat', 'vision_input', 'tool_use', 'json_mode');
        contextWindowSize = 128000; _supportsTools = true; _supportsVision = true;
    } else if (apiModel.id.startsWith('gpt-4-turbo')) {
        capabilities.push('chat', 'tool_use', 'json_mode');
        contextWindowSize = 128000; _supportsTools = true;
        if (apiModel.id.includes('-vision')) { capabilities.push('vision_input'); _supportsVision = true;}
    } else if (apiModel.id.startsWith('gpt-4-32k')) {
        capabilities.push('chat', 'tool_use', 'json_mode');
        contextWindowSize = 32768; _supportsTools = true;
    } else if (apiModel.id.startsWith('gpt-4')) {
        capabilities.push('chat', 'tool_use', 'json_mode');
        contextWindowSize = 8192; _supportsTools = true;
    } else if (apiModel.id.startsWith('gpt-3.5-turbo-16k')) {
        capabilities.push('chat', 'tool_use', 'json_mode');
        contextWindowSize = 16385; _supportsTools = true;
    } else if (apiModel.id.startsWith('gpt-3.5-turbo')) {
        capabilities.push('chat', 'tool_use', 'json_mode');
        // context window for gpt-3.5-turbo can vary (e.g. 4096, 16385 for -0125)
        // Defaulting to a common value, but specific variants might differ.
        contextWindowSize = apiModel.id.includes('0125') || apiModel.id.includes('1106') ? 16385 : 4096;
        _supportsTools = true;
    } else if (apiModel.id.includes('embedding')) {
        capabilities.push('embeddings');
        isEmbeddingModel = true;
        // Common context/input token limit for OpenAI embedding models
        contextWindowSize = 8191; // Max input tokens
    } else {
        // Fallback for other gpt models if any (less common now)
        capabilities.push('chat'); // Assume chat at least
    }

    const pricing = this.resolvePricing(apiModel.id) || { input: 0, output: 0 };
    // modelPricing holds USD per 1K tokens and ModelInfo carries USD per 1M,
    // so every rate is scaled by 1000 here. Cost metering reads modelPricing
    // directly and is unaffected; this conversion is what callers of
    // listAvailableModels() and getModelInfo() see.
    const per1M = (per1K: number) => Math.round(per1K * 1000 * 1e6) / 1e6;

    return {
      modelId: apiModel.id,
      providerId: this.providerId,
      displayName: apiModel.id, // Can be enhanced if more metadata is available
      description: `OpenAI model: ${apiModel.id}`,
      capabilities,
      contextWindowSize,
      pricePer1MTokensInput: per1M(pricing.input),
      pricePer1MTokensOutput: per1M(pricing.output),
      pricePer1MTokensTotal: isEmbeddingModel ? per1M(pricing.input) : undefined,
      supportsStreaming: capabilities.includes('chat'), // Generally, chat models support streaming
      // OpenAI specific details can be added to metadata
      embeddingDimension: apiModel.id.includes('embedding-3-large') ? 3072 :
                          apiModel.id.includes('embedding-3-small') ? 1536 :
                          apiModel.id.includes('ada-002') ? 1536 : undefined,
      lastUpdated: new Date((apiModel.created ?? 0) * 1000).toISOString(),
      status: 'active', // OpenAI doesn't directly provide status in this list, assume active.
    };
  }

  /**
   * Ensures the provider is initialized before use.
   * @private
   * @throws {OpenAIProviderError} If not initialized.
   */
  private ensureInitialized(): void {
    if (!this.isInitialized) {
      throw new OpenAIProviderError(
        'OpenAIProvider is not initialized. Call initialize() first.',
        'PROVIDER_NOT_INITIALIZED'
      );
    }
  }

  /**
   * Resolves the API key to use for a request, prioritizing per-request override, then OAuth flow, then system default.
   * @private
   * @param {string | undefined} apiKeyOverride - API key provided directly in request options.
   * @returns {Promise<string>} The API key to use.
   * @throws {OpenAIProviderError} If no API key is available.
   */
  private async getApiKey(apiKeyOverride?: string): Promise<string> {
    if (apiKeyOverride) return apiKeyOverride;
    if (this.config.oauthFlow) {
      return await this.config.oauthFlow.getAccessToken();
    }
    if (this.keyPool?.hasKeys) return this.keyPool.next();
    if (this.config.apiKey) return this.config.apiKey;
    throw new OpenAIProviderError(
      'No OpenAI API key available for the request.',
      'API_KEY_MISSING'
    );
  }

  /** @inheritdoc */
  public async generateCompletion(
    modelId: string,
    messages: ChatMessage[],
    options: ModelCompletionOptions
  ): Promise<ModelCompletionResponse> {
    this.ensureInitialized();
    const apiKey = await this.getApiKey(options.apiKeyOverride);

    // GPT-5 rejects `reasoning_effort` + function tools on /chat/completions
    // and demands /v1/responses. Route ONLY that exact case there (preserving
    // BOTH reasoning depth and tools); everything else keeps the chat path.
    if (shouldRouteToOpenAiResponsesApi(modelId, messages, options)) {
      const responsesBody = this.buildResponsesPayload(modelId, messages, options);
      const responsesApiResponse = await this.makeApiRequest<OpenAIAPITypes.ResponsesResponse>(
        '/responses',
        'POST',
        apiKey,
        responsesBody,
        false,
        options.requestTimeout
      );
      return this.mapResponsesToCompletionResponse(responsesApiResponse, modelId);
    }

    const requestBody = this.buildChatCompletionPayload(modelId, messages, options, false);

    const apiResponse = await this.makeApiRequest<OpenAIAPITypes.ChatCompletionResponse>(
      '/chat/completions',
      'POST',
      apiKey,
      requestBody,
      false,
      options.requestTimeout
    );

    return this.mapApiToCompletionResponse(apiResponse);
  }

  /** @inheritdoc */
  public async *generateCompletionStream(
    modelId: string,
    messages: ChatMessage[],
    options: ModelCompletionOptions
  ): AsyncGenerator<ModelCompletionResponse, void, undefined> {
    this.ensureInitialized();
    const apiKey = await this.getApiKey(options.apiKeyOverride);

    const requestBody = this.buildChatCompletionPayload(modelId, messages, options, true);

    const stream = (await this.makeApiRequest(
      '/chat/completions',
      'POST',
      apiKey,
      requestBody,
      true, // Indicate streaming response is expected
      options.requestTimeout
    )) as ReadableStream<Uint8Array>;

    // Accumulators for streaming tool calls
    const accumulatedToolCalls: Map<number, { id?: string; type?: 'function'; function?: { name?: string; arguments?: string; } }> = new Map();

    const abortSignal = options.abortSignal;
    if (abortSignal?.aborted) {
      yield {
        id: `openai-abort-${Date.now()}`,
        object: 'chat.completion.chunk',
        created: Math.floor(Date.now()/1000),
        modelId,
        choices: [],
        error: { message: 'Stream aborted prior to first chunk', type: 'abort' },
        isFinal: true,
      };
      return;
    }

    const abortHandler = () => {
      // Emit final abort chunk and ensure generator completes.
      // We cannot directly cancel the underlying ReadableStream cleanly cross-platform here; consumer side will stop.
      // Provide a terminal chunk for consistent teardown.
      // Note: We don't attempt to read further chunks after abort.
    };
    abortSignal?.addEventListener('abort', abortHandler, { once: true });

    for await (const chunk of this.parseSseStream(stream)) {
      if (abortSignal?.aborted) {
        yield {
          id: `openai-abort-${Date.now()}`,
          object: 'chat.completion.chunk',
          created: Math.floor(Date.now()/1000),
          modelId,
          choices: [],
          error: { message: 'Stream aborted by caller', type: 'abort' },
          isFinal: true,
        };
        break;
      }
      if (chunk === '[DONE]') {
        // The [DONE] message is a signal for the end of the stream from OpenAI.
        // A final response chunk with isFinal=true and potential usage should be yielded by the parser if data exists.
        // If OpenAI includes usage in the last data chunk before [DONE], parseSseStream handles it.
        return;
      }
      try {
        const apiChunk = JSON.parse(chunk) as OpenAIAPITypes.ChatCompletionStreamResponse;
        yield this.mapApiToStreamChunkResponse(apiChunk, accumulatedToolCalls);
      } catch (error: unknown) {
        console.warn('OpenAIProvider: Failed to parse stream chunk JSON:', chunk, error);
        // Decide if to yield an error chunk or continue if minor parsing issue
      }
    }

    abortSignal?.removeEventListener('abort', abortHandler);
  }

  /** @inheritdoc */
  public async generateEmbeddings(
    modelId: string,
    texts: string[],
    options?: ProviderEmbeddingOptions
  ): Promise<ProviderEmbeddingResponse> {
    this.ensureInitialized();
    if (!texts || texts.length === 0) {
      throw new OpenAIProviderError('Input texts array cannot be empty for embeddings.', 'EMBEDDING_NO_INPUT');
    }
    const apiKey = await this.getApiKey(options?.apiKeyOverride);

    const payload: Record<string, unknown> = {
      model: modelId,
      input: texts,
    };
    if (options?.encodingFormat) payload.encoding_format = options.encodingFormat;
    if (options?.dimensions) payload.dimensions = options.dimensions;
    if (options?.userId) payload.user = options.userId;
    // OpenAI specific: input_type for their newer models
    if (options?.inputType && (modelId.includes('text-embedding-3') || modelId.includes('ada-002'))) {
        // Not a standard OpenAI param, this is an example if they add it.
        // For now, customModelParams is the way for non-standard things.
    }
    {
      // Strip OpenRouter routing controls and Gemini request fields; see openrouter-only-params.
      const passthrough = stripForeignVendorParams(options?.customModelParams);
      if (passthrough) {
        Object.assign(payload, passthrough);
      }
    }


    const apiResponse = await this.makeApiRequest<OpenAIAPITypes.EmbeddingResponse>(
      '/embeddings',
      'POST',
      apiKey,
      payload
    );

    return this.mapApiToEmbeddingResponse(apiResponse);
  }

  /** @inheritdoc */
  public async listAvailableModels(filter?: { capability?: string }): Promise<ModelInfo[]> {
    this.ensureInitialized();
    // Optionally refresh cache, or rely on initialization. For simplicity, using cached.
    // await this.refreshAvailableModels(); 

    const models = Array.from(this.availableModelsCache.values());
    if (filter?.capability) {
      return models.filter(m => m.capabilities.includes(filter.capability!));
    }
    return models;
  }

  /** @inheritdoc */
  public async getModelInfo(modelId: string): Promise<ModelInfo | undefined> {
    this.ensureInitialized();
    // Could implement a direct fetch if model not in cache, or rely on cached version.
    // For now, uses cached version.
    if (!this.availableModelsCache.has(modelId)) {
        // Attempt to refresh and check again, in case it's a new model
        try {
            await this.refreshAvailableModels();
        } catch (error) {
            console.warn(`OpenAIProvider: Failed to refresh models while fetching info for ${modelId}`, error);
        }
    }
    return this.availableModelsCache.get(modelId);
  }

  /** @inheritdoc */
  public async checkHealth(): Promise<{ isHealthy: boolean; details?: unknown }> {
    try {
      // A lightweight call, e.g., fetching a small model's info or /models with limit=1
      await this.makeApiRequest<OpenAIAPITypes.ModelAPIObject>(
          `/models/${this.defaultModelId || 'gpt-3.5-turbo'}`, // Use a known cheap/fast model
          'GET',
          this.config.apiKey
      );
      return { isHealthy: true };
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Health check failed';
      return { isHealthy: false, details: { message, error } };
    }
  }

  /** @inheritdoc */
  public async shutdown(): Promise<void> {
    // For OpenAIProvider, using stateless HTTP requests, there might not be
    // persistent connections to close. If connection pooling or WebSockets
    // were used, they would be closed here.
    this.isInitialized = false; // Mark as not initialized
    console.log('OpenAIProvider shutdown complete.');
    // No explicit resources to release in this implementation.
  }


  // --- Helper Methods ---

  /**
   * Builds the payload for OpenAI's Chat Completions API.
   * @private
   */
  private buildChatCompletionPayload(
    modelId: string,
    messages: ChatMessage[],
    options: ModelCompletionOptions,
    stream: boolean
  ): Record<string, unknown> {
    const payload: Record<string, unknown> = {
      model: modelId,
      messages: messages.map(m => ({ // Ensure null content is handled if OpenAI expects it explicitly
          role: m.role,
          content: m.content, // OpenAI allows null content for assistant tool_calls message
          name: m.name,
          // Only the standard tool-call fields go on the wire. A tool call can
          // carry provider-specific extras, such as Gemini's thoughtSignature,
          // that the Chat Completions schema does not define.
          tool_calls: m.tool_calls?.map(({ id, type, function: fn }) => ({ id, type, function: fn })),
          tool_call_id: m.tool_call_id,
      })),
      stream: stream,
    };

    // OpenAI's streaming API omits the usage object by default. Without
    // stream_options.include_usage, the final SSE chunk carries finish_reason
    // but no token counts, so downstream `streamText({...}).usage` resolves
    // to zeros. Opting in here gives every streaming caller real usage data
    // (and surfaces a final usage chunk after the [DONE] marker per
    // https://platform.openai.com/docs/api-reference/chat/create#chat-create-stream_options).
    if (stream) {
      payload.stream_options = { include_usage: true };
    }

    // Reasoning models (o-series, GPT-5) reject custom sampling params with an
    // HTTP 400 — they run only at their fixed defaults. Omit temperature/top_p
    // for them, mirroring the max_tokens → max_completion_tokens switch below.
    if (!isOpenAIReasoningModel(modelId)) {
      if (options.temperature !== undefined) payload.temperature = options.temperature;
      if (options.topP !== undefined) payload.top_p = options.topP;
    } else {
      // Reasoning models (o-series, GPT-5) take `reasoning_effort`
      // (none|low|medium|high|xhigh) in place of the sampling params they
      // reject. Map the agentos effort scale onto it so `effort: 'max'` actually
      // drives gpt-5.x at its xhigh ceiling instead of being silently dropped.
      // customModelParams (below) can still override.
      // GPT-5 rejects `reasoning_effort` + function tools on chat/completions
      // (it demands the Responses API, which agentos doesn't implement). When
      // the request carries tools, DROP the effort param instead of 400ing —
      // the model reasons at its default depth and the tool call succeeds.
      const reasoningEffort = mapEffortToOpenAiReasoningEffortForModel(options.effort, modelId);
      const dropEffortForTools =
        requestHasFunctionTools(options.tools) &&
        openAiRejectsReasoningEffortWithTools(modelId);
      if (reasoningEffort !== undefined && !dropEffortForTools) {
        payload.reasoning_effort = reasoningEffort;
      }
    }
    if (options.maxTokens !== undefined) {
      // Clamp to the model's real output ceiling first: a request sized for
      // a flagship model (e.g. 32000 for Claude Opus) is rejected by gpt-4o
      // (max 16384) when the fallback chain routes here. The clamp is a
      // no-op for models whose ceiling is at or above the request.
      const effMaxTokens = clampMaxOutputTokens(modelId, options.maxTokens) ?? options.maxTokens;
      // OpenAI's reasoning + GPT-5 model families reject `max_tokens`
      // and require `max_completion_tokens` instead. Legacy chat
      // completions (gpt-4o, gpt-4-turbo, gpt-3.5, etc.) keep
      // `max_tokens`. The two fields are otherwise equivalent: same
      // semantic meaning, just renamed in the newer API surface.
      if (modelRequiresMaxCompletionTokens(modelId)) {
        payload.max_completion_tokens = effMaxTokens;
      } else {
        payload.max_tokens = effMaxTokens;
      }
    }
    if (options.presencePenalty !== undefined) payload.presence_penalty = options.presencePenalty;
    if (options.frequencyPenalty !== undefined) payload.frequency_penalty = options.frequencyPenalty;
    if (options.stopSequences !== undefined) payload.stop = options.stopSequences;
    if (options.userId !== undefined) payload.user = options.userId;
    if (options.tools !== undefined) payload.tools = options.tools;
    if (options.toolChoice !== undefined) payload.tool_choice = options.toolChoice;
    // Coerce to an OpenAI-valid response_format, or drop it. agentos builds a
    // provider-specific responseFormat for the PRIMARY provider and reuses it
    // across the fallback chain; an Anthropic tool-marker / Gemini-internal
    // object reaching OpenAI verbatim makes the API reject the request
    // ("Missing required parameter: 'response_format.type'"). Mirrors the
    // guard OpenRouterProvider already applies. See openai-response-format-guard.ts.
    if (options.responseFormat !== undefined) {
      const coerced = toOpenAiResponseFormat(options.responseFormat);
      if (coerced !== undefined) payload.response_format = coerced;
    }

    // Typed OpenAI prompt-cache / service-tier policy (spec batch-1 C2).
    // customModelParams (below) keeps last-write precedence — it is the
    // raw escape hatch and intentionally overrides these typed fields.
    this.applyCacheAndTierParams(payload, modelId, options);

    {
      // Strip OpenRouter routing controls and Gemini request fields; see openrouter-only-params.
      const passthrough = stripForeignVendorParams(options.customModelParams);
      if (passthrough) {
        Object.assign(payload, passthrough);
      }
    }

    return payload;
  }

  /**
   * True when requests target api.openai.com rather than an
   * OpenAI-compatible gateway (the Groq/xAI/Together/Mistral wrappers pass
   * their own baseURL). Gates zero-config defaults that emit request
   * fields some gateways reject, so the match is an exact parsed-hostname
   * comparison — a substring test would classify look-alike gateway hosts
   * (api.openai.com.example.net) as native. Tolerates an uninitialized
   * provider (payload builders are unit-tested without initialize()) and
   * malformed URLs: anything unparseable is not native, never a throw.
   * @private
   */
  private isNativeOpenAiEndpoint(): boolean {
    try {
      const url = new URL(this.config?.baseURL ?? '');
      return (
        url.protocol === 'https:' &&
        url.hostname.toLowerCase() === 'api.openai.com' &&
        (url.port === '' || url.port === '443')
      );
    } catch {
      return false;
    }
  }

  /**
   * Apply the typed prompt-cache key/retention and service-tier options to
   * an outgoing payload (Chat and Responses builders share this; spec
   * batch-1 C2). Unsupported retention combinations are omitted fail-closed
   * with a debug log — never a hard error.
   * @private
   */
  private applyCacheAndTierParams(
    payload: Record<string, unknown>,
    modelId: string,
    options: ModelCompletionOptions,
  ): void {
    // Zero-config default: on the native OpenAI endpoint the shard key
    // derives from the session id ('auto' semantics) unless the caller
    // opted out via promptCacheKey or cache: false. OpenAI-compatible
    // gateways (Groq/xAI/Together/Mistral wrappers pass their own baseURL)
    // keep the omit default — some reject unknown request fields.
    const defaultCacheKeyMode =
      options.cache !== false && this.isNativeOpenAiEndpoint() ? ('auto' as const) : undefined;
    const cacheKey = resolvePromptCacheKey(
      options.promptCacheKey ?? defaultCacheKeyMode,
      options.promptCacheSessionId ?? options.sessionId,
    );
    if (cacheKey) payload.prompt_cache_key = cacheKey;
    if (options.promptCacheRetention) {
      const retention = resolveOpenAiCacheRetentionParams(modelId, options.promptCacheRetention);
      if (retention) {
        Object.assign(payload, retention);
      } else {
        console.debug(
          `OpenAIProvider: promptCacheRetention '${options.promptCacheRetention}' unsupported on ${modelId}; omitted`,
        );
      }
    }
    if (options.serviceTier) payload.service_tier = options.serviceTier;
  }

  /**
   * Build a `/v1/responses` request body for a gpt-5/gpt-6 reasoning +
   * function-tool call (the only case
   * {@link shouldRouteToOpenAiResponsesApi} routes here).
   * Preserves BOTH `reasoning.effort` and `tools`, which chat/completions
   * rejects together. No structured output by construction (the router
   * excludes responseFormat). Message content arrives as either a plain
   * string or an all-text block array — {@link flattenResponsesTextContent}
   * joins the latter, so cache-marked `systemBlocks` survive intact instead of
   * collapsing to an empty prompt. True multimodal content never reaches here
   * (the router excludes it via {@link isResponsesMappableContent}).
   * Live-probed 2026-07-08; block-content path live-probed 2026-09-12.
   * @private
   */
  private buildResponsesPayload(
    modelId: string,
    messages: ChatMessage[],
    options: ModelCompletionOptions
  ): Record<string, unknown> {
    const input: Array<Record<string, unknown>> = [];
    for (const m of messages) {
      const text = flattenResponsesTextContent(m.content);
      if (m.role === 'tool') {
        // Tool result → function_call_output paired by call_id.
        input.push({ type: 'function_call_output', call_id: m.tool_call_id, output: text });
        continue;
      }
      if (m.role === 'assistant' && m.tool_calls && m.tool_calls.length > 0) {
        // Optional assistant text first, then one function_call item per call.
        if (text.length > 0) input.push({ role: 'assistant', content: text });
        for (const tc of m.tool_calls) {
          input.push({
            type: 'function_call',
            call_id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
          });
        }
        continue;
      }
      // system / user / plain assistant. Drop an empty assistant turn (no text,
      // no tool_calls carries nothing); keep empty system/user defensively.
      if (m.role === 'assistant' && text.length === 0) continue;
      input.push({ role: m.role, content: text });
    }

    const payload: Record<string, unknown> = {
      model: modelId,
      input,
      tools: this.toResponsesTools(options.tools),
      reasoning: { effort: mapEffortToOpenAiResponsesEffort(modelId, options.effort) },
      store: false,
    };
    if (options.toolChoice !== undefined) {
      payload.tool_choice = this.toResponsesToolChoice(options.toolChoice);
    }
    if (options.maxTokens !== undefined) {
      payload.max_output_tokens = clampMaxOutputTokens(modelId, options.maxTokens) ?? options.maxTokens;
    }
    if (options.userId !== undefined) payload.safety_identifier = options.userId;
    // presence/frequency penalties + stop are intentionally OMITTED — reasoning
    // models don't meaningfully use them and the orchestrator sets none;
    // temperature/top_p omitted (reasoning models reject them). See spec §D3.

    // Typed prompt-cache / service-tier policy — same helper as the Chat
    // builder; customModelParams below keeps last-write precedence.
    this.applyCacheAndTierParams(payload, modelId, options);

    {
      // Strip OpenRouter routing controls and Gemini request fields; see openrouter-only-params.
      // The /responses body rejects unknown top-level fields just like
      // /chat/completions.
      const passthrough = stripForeignVendorParams(options.customModelParams);
      if (passthrough) {
        Object.assign(payload, passthrough);
      }
    }
    return payload;
  }

  /** Reshape chat-completions function tools → the Responses flat tool shape. */
  private toResponsesTools(tools: unknown): unknown {
    if (!Array.isArray(tools)) return tools;
    return tools.map((t) => {
      const tool = t as { type?: string; function?: { name?: string; description?: string; parameters?: unknown }; name?: string };
      if (tool?.type === 'function' && tool.function) {
        return {
          type: 'function',
          name: tool.function.name,
          description: tool.function.description,
          parameters: tool.function.parameters,
          strict: false,
        };
      }
      return t; // already-flat or non-function tool: pass through
    });
  }

  /** Map chat-completions tool_choice → the Responses tool_choice shape. */
  private toResponsesToolChoice(toolChoice: unknown): unknown {
    if (typeof toolChoice === 'string') return toolChoice; // 'auto' | 'required' | 'none'
    const tc = toolChoice as { type?: string; function?: { name?: string } };
    if (tc?.type === 'function' && tc.function?.name) {
      return { type: 'function', name: tc.function.name };
    }
    return toolChoice;
  }

  /**
   * Map a `/v1/responses` response into the same {@link ModelCompletionResponse}
   * shape the rest of agentos consumes from chat/completions: assistant text
   * from `output_text` message parts, tool calls from `function_call` items
   * (call_id → id).
   *
   * An output array with NO items at all THROWS (→ fallback advances) rather
   * than returning an empty success (Codex-Medium-2). Output that holds items
   * but no `message`/`function_call` does NOT throw: a reasoning model returns
   * reasoning-only output when reasoning exhausts the output budget, which is
   * a legitimate `length` finish. It maps to an empty turn (`content: null`)
   * carrying the real finishReason, so the caller can retry with a larger
   * budget instead of losing a whole multi-step build to one capped turn.
   * @private
   */
  private mapResponsesToCompletionResponse(
    apiResponse: OpenAIAPITypes.ResponsesResponse,
    modelId: string
  ): ModelCompletionResponse {
    const output = apiResponse.output ?? [];
    let assistantText = '';
    const toolCalls: Array<{ id: string; type: 'function'; function: { name: string; arguments: string } }> = [];
    for (const item of output) {
      if (item.type === 'message') {
        for (const part of item.content ?? []) {
          if (part.type === 'output_text' && typeof part.text === 'string') assistantText += part.text;
        }
      } else if (item.type === 'function_call') {
        toolCalls.push({
          id: item.call_id ?? '',
          type: 'function',
          function: { name: item.name ?? '', arguments: item.arguments ?? '' },
        });
      }
      // reasoning + any other item type: ignored.
    }

    const incompleteReason = apiResponse.incomplete_details?.reason;
    const finishReason = toolCalls.length > 0
      ? 'tool_calls'
      : apiResponse.status === 'incomplete'
        ? (incompleteReason === 'max_output_tokens' ? 'length'
          : incompleteReason === 'content_filter' ? 'content_filter'
          : 'stop')
        : 'stop';

    if (output.length === 0) {
      // GENUINELY empty 2xx — no output items at all. This is the malformed
      // body the guard was written for: throw so generateText's fallback
      // advances instead of treating it as a successful empty turn.
      //
      // Deliberately NOT triggered by "items present but none usable": a
      // reasoning model returns output holding ONLY a `reasoning` item when
      // reasoning consumes the whole output budget. That is a well-formed
      // `length` finish, not a malformed body, and this check used to sit
      // ABOVE the finishReason computation — so it threw before the code that
      // already knew how to express it could run, hard-erroring entire builds
      // over one budget-capped turn.
      throw new OpenAIProviderError(
        `OpenAI /responses returned no usable output (status: ${apiResponse.status ?? 'unknown'})`,
        'INVALID_RESPONSE',
        undefined,
        undefined,
        undefined,
        apiResponse as unknown as Record<string, unknown>
      );
    }

    if (assistantText.length === 0 && toolCalls.length === 0) {
      // Items present but no message/function_call — reasoning-only output.
      // Surfaced as an empty turn carrying the real finishReason ('length'
      // when max_output_tokens capped it), so the caller can retry with a
      // larger budget. Reasoning tokens are drawn from the SAME output budget
      // as the visible answer, so a reasoning model needs a materially larger
      // maxTokens than a non-reasoning one to leave room for a reply.
      console.warn(
        `OpenAIProvider: /responses returned reasoning-only output for ${apiResponse.model ?? modelId} ` +
        `(status: ${apiResponse.status ?? 'unknown'}${incompleteReason ? `, reason: ${incompleteReason}` : ''}` +
        `${apiResponse.usage?.output_tokens !== undefined ? `, output_tokens: ${apiResponse.usage.output_tokens}` : ''}` +
        `) — returning an empty turn with finishReason '${finishReason}'. ` +
        `Raise max output tokens: reasoning tokens consume the same budget as the reply.`
      );
    }

    return {
      id: apiResponse.id,
      object: 'chat.completion',
      created: apiResponse.created_at ?? Math.floor(Date.now() / 1000),
      modelId: apiResponse.model ?? modelId,
      ...(typeof apiResponse.service_tier === 'string' ? { serviceTier: apiResponse.service_tier } : {}),
      choices: [{
        index: 0,
        message: {
          role: 'assistant' as ModelCompletionChoice['message']['role'],
          content: assistantText.length > 0 ? assistantText : null,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        },
        finishReason,
        logprobs: null,
      }],
      usage: apiResponse.usage
        ? this.calculateUsage(
            {
              prompt_tokens: apiResponse.usage.input_tokens,
              completion_tokens: apiResponse.usage.output_tokens,
              total_tokens: apiResponse.usage.total_tokens,
              // Responses API spells the cached detail input_tokens_details;
              // re-key to the chat spelling the shared mapper reads.
              ...(apiResponse.usage.input_tokens_details
                ? { prompt_tokens_details: {
                      cached_tokens: apiResponse.usage.input_tokens_details.cached_tokens,
                      cache_write_tokens: apiResponse.usage.input_tokens_details.cache_write_tokens,
                    } }
                : {}),
            },
            apiResponse.model ?? modelId
          )
        : undefined,
    };
  }

  /**
   * Maps the raw OpenAI API Chat Completion response to the standard ModelCompletionResponse.
   * @private
   */
  private mapApiToCompletionResponse(
    apiResponse: OpenAIAPITypes.ChatCompletionResponse
  ): ModelCompletionResponse {
    return {
      id: apiResponse.id,
      object: apiResponse.object,
      created: apiResponse.created,
      modelId: apiResponse.model,
      choices: (apiResponse.choices ?? []).map(c => ({
        index: c.index,
        message: {
          role: c.message.role as ModelCompletionChoice['message']['role'],
          content: c.message.content,
          tool_calls: c.message.tool_calls,
        },
        finishReason: c.finish_reason,
        logprobs: c.logprobs,
      })),
      usage: apiResponse.usage ? this.calculateUsage(apiResponse.usage, apiResponse.model) : undefined,
      ...(typeof apiResponse.service_tier === 'string' ? { serviceTier: apiResponse.service_tier } : {}),
      // No deltas for non-streaming response
    };
  }

    /**
     * Maps an OpenAI API stream chunk to a ModelCompletionResponse chunk.
     * Handles accumulation of tool calls.
     * @private
     */
    private mapApiToStreamChunkResponse(
        apiChunk: OpenAIAPITypes.ChatCompletionStreamResponse,
        accumulatedToolCalls: Map<number, { id?: string; type?: 'function'; function?: { name?: string; arguments?: string; } }>
    ): ModelCompletionResponse {
        const choice = apiChunk.choices?.[0];
        let responseTextDelta: string | undefined;
        let toolCallsDeltas: ModelCompletionResponse['toolCallsDeltas'];
        let finalUsage: ModelUsage | undefined;

        // With stream_options.include_usage, OpenAI emits a trailing chunk with
        // an empty `choices` array and a populated `usage` object. That chunk
        // carries no content / no finish_reason — its only job is to deliver
        // token totals. Treat it as a final usage-only chunk so callers see
        // accurate token counts after the stream resolves.
        if ((!apiChunk.choices || apiChunk.choices.length === 0) && apiChunk.usage) {
            const usageOnlyUsage = this.calculateUsage(apiChunk.usage, apiChunk.model);
            return {
                id: apiChunk.id,
                object: apiChunk.object,
                created: apiChunk.created,
                modelId: apiChunk.model,
                choices: [],
                isFinal: true,
                usage: usageOnlyUsage,
                ...(typeof apiChunk.service_tier === 'string'
                  ? { serviceTier: apiChunk.service_tier }
                  : {}),
            };
        }

        // After the usage-only sentinel above, a healthy chunk always carries a
        // choice. Guard defensively so the non-optional accesses below
        // (choice.index, choice.delta.role) can't crash on a malformed chunk.
        if (!choice) {
            return {
                id: apiChunk.id,
                object: apiChunk.object,
                created: apiChunk.created,
                modelId: apiChunk.model,
                choices: [],
            };
        }

        if (choice?.delta?.content) {
            responseTextDelta = choice.delta.content;
        }

        if (choice?.delta?.tool_calls) {
            toolCallsDeltas = [];
            choice.delta.tool_calls.forEach(tcDelta => {
                let currentToolCall = accumulatedToolCalls.get(tcDelta.index);
                if (!currentToolCall) {
                    currentToolCall = { function: {} }; // Initialize if new
                    if (tcDelta.id) currentToolCall.id = tcDelta.id;
                    if (tcDelta.type) currentToolCall.type = tcDelta.type;
                }

                if (tcDelta.id && !currentToolCall.id) currentToolCall.id = tcDelta.id; // Ensure ID is set if provided in delta
                if (tcDelta.function?.name) currentToolCall.function!.name = (currentToolCall.function!.name || '') + tcDelta.function.name;
                if (tcDelta.function?.arguments) currentToolCall.function!.arguments = (currentToolCall.function!.arguments || '') + tcDelta.function.arguments;
                
                accumulatedToolCalls.set(tcDelta.index, currentToolCall);

                toolCallsDeltas!.push({
                    index: tcDelta.index,
                    id: tcDelta.id,
                    type: tcDelta.type,
                    function: tcDelta.function ? {
                        name: tcDelta.function.name,
                        arguments_delta: tcDelta.function.arguments
                    } : undefined,
                });
            });
        }
        
        const isFinal = !!choice?.finish_reason;
        if (isFinal) {
            // OpenAI might send usage in the last chunk or not at all for streams.
            // If `apiChunk.usage` exists (it's optional in stream chunks per some docs), use it.
            if (apiChunk.usage) {
                 finalUsage = this.calculateUsage(apiChunk.usage, apiChunk.model);
            }
        }
        
        // Construct the primary choice message for the response
        const responseChoice: ModelCompletionChoice = {
            index: choice.index,
            message: {
                role: (choice.delta.role || 'assistant') as ModelCompletionChoice['message']['role'], // Role usually comes once or is 'assistant'
                content: responseTextDelta || null, // Content is delta
                 // For final chunk, assemble complete tool_calls
                tool_calls: isFinal ? Array.from(accumulatedToolCalls.values()).map(accTc => ({
                    id: accTc.id!, // ID should be present by now
                    type: accTc.type!, // Type should be present
                    function: {
                        name: accTc.function!.name!,
                        arguments: accTc.function!.arguments!
                    }
                })).filter(tc => tc.id && tc.function.name) : undefined, // Only include if valid
            },
            finishReason: choice.finish_reason || null,
            logprobs: choice.logprobs,
        };


        return {
            id: apiChunk.id,
            object: apiChunk.object,
            created: apiChunk.created,
            modelId: apiChunk.model,
            choices: [responseChoice],
            responseTextDelta,
            toolCallsDeltas,
            isFinal,
            usage: finalUsage, // Only on the very final message from the stream
        };
    }


  /**
   * Maps the raw OpenAI API Embedding response to the standard ProviderEmbeddingResponse.
   * @private
   */
  private mapApiToEmbeddingResponse(
    apiResponse: OpenAIAPITypes.EmbeddingResponse
  ): ProviderEmbeddingResponse {
    return {
      object: 'list',
      data: apiResponse.data.map(d => ({
        object: 'embedding',
        embedding: d.embedding,
        index: d.index,
      })),
      model: apiResponse.model,
      usage: {
        prompt_tokens: apiResponse.usage.prompt_tokens,
        total_tokens: apiResponse.usage.total_tokens,
        costUSD: this.calculateCost(
          apiResponse.usage.prompt_tokens,
          0, // No completion tokens for embeddings
          apiResponse.model,
          true // isEmbedding
        ),
      },
    };
  }

  /**
   * Calculates token usage and cost.
   * @private
   */
  private calculateUsage(
    usage: {
      prompt_tokens: number;
      completion_tokens: number;
      total_tokens: number;
      prompt_tokens_details?: { cached_tokens?: number; cache_write_tokens?: number };
    },
    modelId: string
  ): ModelUsage {
    // OpenAI automatic caching reports cached prompt tokens as detail
    // fields (reads AND — on GPT-5.6+, billed 1.25x — writes; both remain
    // INCLUDED in prompt_tokens, unlike Anthropic's exclusive accounting).
    // Normalize into cacheReadInputTokens / cacheCreationInputTokens so
    // caching stops being invisible in platform telemetry; cost stays
    // computed off the full prompt_tokens (the discount/premium is a
    // billing-side rate, not a token-count change). Because prompt_tokens
    // is already inclusive here, inclusiveInputTokens is prompt_tokens
    // as-is (Anthropic computes input + cache_read + cache_creation).
    const cachedTokens = usage.prompt_tokens_details?.cached_tokens;
    const cacheWriteTokens = usage.prompt_tokens_details?.cache_write_tokens;
    return {
      promptTokens: usage.prompt_tokens,
      completionTokens: usage.completion_tokens,
      totalTokens: usage.total_tokens,
      costUSD: this.calculateCost(usage.prompt_tokens, usage.completion_tokens, modelId),
      ...(typeof usage.prompt_tokens === 'number'
        ? { inclusiveInputTokens: usage.prompt_tokens }
        : {}),
      ...(typeof cachedTokens === 'number' && cachedTokens >= 0
        ? { cacheReadInputTokens: cachedTokens }
        : {}),
      ...(typeof cacheWriteTokens === 'number' && cacheWriteTokens >= 0
        ? { cacheCreationInputTokens: cacheWriteTokens }
        : {}),
    };
  }

  /**
   * Price-table row for a model id. OpenAI responses name the dated snapshot
   * that served the call (`gpt-4.1-2025-04-14` for a `gpt-4.1` request), and
   * usage is costed from that response id, so an exact miss retries with a
   * trailing `-YYYY-MM-DD` removed. An exact row still wins, which keeps
   * snapshots priced apart from their alias (`gpt-4o-2024-05-13`,
   * `o3-pro-2025-06-10`).
   *
   * @param modelId Model id as requested or as echoed by the API.
   * @returns USD per 1K tokens, or undefined when no row matches.
   */
  private resolvePricing(modelId: string | undefined): { input: number; output: number } | undefined {
    if (!modelId) return undefined;
    return this.modelPricing[modelId] ?? this.modelPricing[modelId.replace(/-\d{4}-\d{2}-\d{2}$/, '')];
  }

  /**
   * Estimated USD cost of one API call, from the per-1K rates in
   * `modelPricing`. When the model has long-context pricing
   * ({@link openAiHasLongContextPricing}) and the prompt exceeds
   * {@link OPENAI_LONG_CONTEXT_THRESHOLD_TOKENS} input tokens, the whole call
   * bills at the long-context rates: every input token at 2x and every output
   * token at 1.5x.
   *
   * @param promptTokens Input tokens, cached tokens included. OpenAI compares
   *   this same count against the long-context threshold.
   * @param completionTokens Output tokens, reasoning tokens included.
   * @param modelId Model id as echoed by the API; a dated snapshot without its
   *   own row resolves to its base row.
   * @param isEmbedding Prices `promptTokens` alone at the embedding rate.
   * @returns Cost in USD, or undefined when the model has no price row.
   * @private
   */
  private calculateCost(
    promptTokens: number,
    completionTokens: number,
    modelId: string,
    isEmbedding: boolean = false
  ): number | undefined {
    const pricing = this.resolvePricing(modelId);
    if (!pricing) return undefined; // Pricing info not available

    if (isEmbedding) {
      return (promptTokens / 1000) * pricing.input;
    }
    // A long prompt moves the output onto the higher rate too, because
    // OpenAI prices the full request at the long-context tier.
    const isLong =
      promptTokens > OPENAI_LONG_CONTEXT_THRESHOLD_TOKENS && openAiHasLongContextPricing(modelId);
    const inputRate = isLong ? pricing.input * OPENAI_LONG_CONTEXT_INPUT_MULTIPLIER : pricing.input;
    const outputRate = isLong ? pricing.output * OPENAI_LONG_CONTEXT_OUTPUT_MULTIPLIER : pricing.output;
    return (promptTokens / 1000) * inputRate + (completionTokens / 1000) * outputRate;
  }

  /**
   * Makes an API request to OpenAI with error handling and retries.
   * @private
   * @template T The expected response type.
   * @param {string} endpoint - The API endpoint (e.g., "/chat/completions").
   * @param {'GET' | 'POST'} method - HTTP method.
   * @param {string} apiKey - The API key to use.
   * @param {Record<string, unknown>} [body] - The request body for POST requests.
   * @param {boolean} [expectStream] - Whether the response is expected to be a stream.
   * @returns {Promise<T | ReadableStream<Uint8Array>>} The API response or stream.
   * @throws {OpenAIProviderError} If the request fails after retries or for non-retryable errors.
   */
  private async makeApiRequest<T = unknown>(
    endpoint: string,
    method: 'GET' | 'POST',
    apiKey: string,
    body?: Record<string, unknown>,
    expectStream?: false,
    requestTimeoutOverride?: number
  ): Promise<T>;
  private async makeApiRequest(
    endpoint: string,
    method: 'GET' | 'POST',
    apiKey: string,
    body: Record<string, unknown> | undefined,
    expectStream: true,
    requestTimeoutOverride?: number
  ): Promise<ReadableStream<Uint8Array>>;
  private async makeApiRequest<T = unknown>(
    endpoint: string,
    method: 'GET' | 'POST',
    apiKey: string,
    body?: Record<string, unknown>,
    expectStream: boolean = false,
    requestTimeoutOverride?: number
  ): Promise<T | ReadableStream<Uint8Array>> {
    const url = `${this.config.baseURL}${endpoint}`;
    const headers: Record<string, string> = {
      'Authorization': `Bearer ${apiKey}`,
      'User-Agent': 'AgentOS/1.0 (OpenAIProvider)',
      ...(this.config.customHeaders || {}),
    };
    if (method === 'POST') {
      headers['Content-Type'] = 'application/json';
    }
    if (this.config.organizationId) {
      headers['OpenAI-Organization'] = this.config.organizationId;
    }

    let lastError: Error = new OpenAIProviderError('Request failed after all retries.', 'MAX_RETRIES_REACHED');

    for (let attempt = 0; attempt < this.config.maxRetries!; attempt++) {
      const controller = new AbortController();
      // CR8: honor a per-call requestTimeout override (e.g. long codegen) over
      // the provider's default; only Anthropic read this before.
      const effectiveTimeout =
        typeof requestTimeoutOverride === 'number' && requestTimeoutOverride > 0
          ? requestTimeoutOverride
          : this.config.requestTimeout;
      const timeoutId = setTimeout(() => controller.abort(), effectiveTimeout);

      try {
        const requestOptions: RequestInit = {
          method,
          headers,
          body: body ? JSON.stringify(body) : undefined,
          signal: controller.signal,
        };

        const response = await fetch(url, requestOptions);
        clearTimeout(timeoutId);

        if (!response.ok) {
          const errorData: OpenAIAPITypes.APIErrorResponse = await response.json().catch(() => ({}));
          const errorMessage = errorData.error?.message || `HTTP error ${response.status}: ${response.statusText}`;
          const errorType = errorData.error?.type;
          const errorCode = errorData.error?.code;
          
          // Non-retryable errors
          if (response.status === 401 || response.status === 403 || response.status === 400 || response.status === 404) {
            throw new OpenAIProviderError(errorMessage, 'API_CLIENT_ERROR', errorCode || undefined, errorType, response.status, errorData);
          }
          // For other server errors or rate limits, prepare for retry
          lastError = new OpenAIProviderError(errorMessage, 'API_SERVER_ERROR', errorCode || undefined, errorType, response.status, errorData);
          if (response.status === 429) { // Rate limit
            lastError = new OpenAIProviderError(errorMessage, 'RATE_LIMIT_EXCEEDED', errorCode || undefined, errorType, response.status, errorData);
            const retryAfter = response.headers.get('retry-after'); // seconds
            // Retry-After is authoritative when present; otherwise jittered backoff.
            const retryAfterMs = retryAfter ? parseInt(retryAfter, 10) * 1000 : computeRetryBackoffMs(attempt);
            await new Promise(resolve => setTimeout(resolve, retryAfterMs));
            continue; // Retry
          }
          // Other retryable server errors
          if (response.status >= 500) {
            await new Promise(resolve => setTimeout(resolve, computeRetryBackoffMs(attempt))); // jittered exponential backoff
            continue; // Retry
          }
          throw lastError; // If not explicitly handled for retry, throw
        }

        if (expectStream) {
          if (!response.body) {
            throw new OpenAIProviderError('Expected a stream response but body was null.', 'STREAM_BODY_NULL');
          }
          return response.body;
        }
        return (await response.json()) as T;

      } catch (error: unknown) {
        clearTimeout(timeoutId);
        if (error instanceof OpenAIProviderError) { // If already our custom error, re-throw if not retryable
             if (error.code === 'API_CLIENT_ERROR') throw error;
             lastError = error;
        } else if (error instanceof Error && error.name === 'AbortError') {
          lastError = new OpenAIProviderError(`Request timed out after ${effectiveTimeout}ms.`, 'REQUEST_TIMEOUT', undefined, undefined, undefined, error);
        } else {
          // fetch quotes a URL it rejects (base URL credentials included) and
          // a header value it rejects (the key), so the message is masked and
          // the raw error is not kept.
          lastError = new OpenAIProviderError(
            redactUrlSecrets(error instanceof Error ? error.message : 'Network or unknown error', this.config.baseURL, [apiKey]),
            'NETWORK_ERROR',
          );
        }
        
        if (attempt === this.config.maxRetries! - 1) {
          break; // Last attempt failed, break to throw lastError
        }
        // Equal-jitter exponential backoff for retries not handled by 429
        const delay = computeRetryBackoffMs(attempt);
        // Keep retry logs concise — network errors are common and the stack trace is noise
        const isNetwork = lastError instanceof OpenAIProviderError && lastError.code === 'NETWORK_ERROR';
        const shortMsg = isNetwork ? 'Network unreachable' : lastError.message;
        console.warn(`[LLM] Retry ${attempt + 1}/${this.config.maxRetries! - 1} in ${(delay / 1000).toFixed(1)}s — ${shortMsg}`);
        await new Promise(resolve => setTimeout(resolve, delay));
      }
    }
    // For network errors, replace the cryptic cause chain with a clean message
    if (lastError instanceof OpenAIProviderError && lastError.code === 'NETWORK_ERROR') {
      throw new OpenAIProviderError(
        redactUrlSecrets(
          `Network error: unable to reach ${this.config.baseURL}. Check your internet connection.`,
          this.config.baseURL,
          [apiKey],
        ),
        'NETWORK_ERROR',
      );
    }
    throw lastError; // Throw the last encountered error after all retries
  }

  /**
   * Parses an SSE (Server-Sent Events) stream.
   * @private
   */
  private async *parseSseStream(stream: ReadableStream<Uint8Array>): AsyncGenerator<string, void, undefined> {
    const reader = stream.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        
        let eolIndex;
        // An SSE event is typically defined by "data: ...\n\n"
        // Or multiple "data: ..." lines followed by \n\n
        while ((eolIndex = buffer.indexOf('\n\n')) >= 0) {
            const messageBlock = buffer.substring(0, eolIndex);
            buffer = buffer.substring(eolIndex + 2); // +2 for \n\n

            const lines = messageBlock.split('\n');
            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    const dataContent = line.substring('data: '.length).trim();
                    if (dataContent) { // Ensure not empty data string
                        yield dataContent;
                    }
                }
            }
        }
      }
      // Process any remaining data in the buffer (e.g. if stream ends without \n\n)
      if (buffer.startsWith('data: ')) {
        const dataContent = buffer.substring('data: '.length).trim();
        if (dataContent) yield dataContent;
      } else if (buffer.trim()) { // Sometimes final [DONE] might not be prefixed by "data: "
          if (buffer.trim() === "[DONE]") yield "[DONE]";
          else console.warn("OpenAIProvider: Trailing stream data not processed:", buffer);
      }

    } catch (error: unknown) {
        const message = error instanceof Error ? error.message : 'Stream parsing error';
        console.error('OpenAIProvider: Error reading or parsing SSE stream:', message, error);
        throw new OpenAIProviderError(message, 'STREAM_PARSING_ERROR', undefined, undefined, undefined, error);
    } finally {
      // Ensure stream is closed if `break` or `return` is called within the generator
      if (!reader.closed) {
          await reader.cancel().catch(err => console.error("OpenAIProvider: Error cancelling stream reader:", err));
      }
      // console.log("OpenAIProvider: SSE Stream parsing finished.");
    }
  }
}
