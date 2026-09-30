/**
 * @file generateText-anthropic-wire.integration.test.ts
 * generateText, streamText and generateObject against the real
 * AnthropicProvider, with only fetch stubbed. The stub answers per model, read
 * from each request body, so a fallback hop to a second Claude model gets its
 * own reply. The tests assert on what the public API returns and on the
 * request bodies the provider put on the wire.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { z } from 'zod';

const fetchMock = vi.fn();
vi.stubGlobal('fetch', fetchMock);

import { generateText } from '../../generateText.js';
import { generateObject } from '../../generateObject.js';
import { globalLLMProviderHealth } from '../../../core/safety/LLMProviderHealthRegistry.js';

type Json = Record<string, any>;
type Reply = () => Response;

/** An SSE response carrying `events` in order. */
function sse(events: unknown[]): Response {
  return new Response(events.map((e) => `data: ${JSON.stringify(e)}\n\n`).join(''), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });
}

/** The message_start event of a streamed reply from `model`. */
function messageStart(model: string, usage: Json): Json {
  return {
    type: 'message_start',
    message: {
      id: `msg_${model}`, type: 'message', role: 'assistant', content: [], model,
      stop_reason: null, stop_sequence: null, usage: { output_tokens: 1, ...usage },
    },
  };
}

/** A completed text reply from `model`. */
function textTurn(model: string, text: string, usage: Json = { input_tokens: 10 }, outputTokens = 3): Reply {
  return () =>
    sse([
      messageStart(model, usage),
      { type: 'content_block_start', index: 0, content_block: { type: 'text', text: '' } },
      { type: 'content_block_delta', index: 0, delta: { type: 'text_delta', text } },
      { type: 'content_block_stop', index: 0 },
      {
        type: 'message_delta',
        delta: { stop_reason: 'end_turn', stop_sequence: null },
        usage: { output_tokens: outputTokens },
      },
      { type: 'message_stop' },
    ]);
}

/** Routes each POST to the next reply queued for the model its body names. */
function route(replies: Record<string, Reply[]>): void {
  fetchMock.mockImplementation(async (_url: unknown, init?: { body?: unknown }) => {
    const body = JSON.parse(String(init?.body)) as Json;
    const next = replies[String(body.model)]?.shift();
    if (!next) throw new Error(`unexpected request for model ${String(body.model)}`);
    return next();
  });
}

/** Parsed body of every request, in call order. */
function postedBodies(): Json[] {
  return fetchMock.mock.calls.map(([, init]) => JSON.parse(String((init as { body?: unknown }).body)) as Json);
}

const savedAnthropicKey = process.env.ANTHROPIC_API_KEY;
const savedOpenRouterKey = process.env.OPENROUTER_API_KEY;

beforeAll(() => {
  // Fallback hops resolve their key from the environment.
  process.env.ANTHROPIC_API_KEY = 'sk-ant-wire-test';
  // Without an Anthropic key the resolver reroutes to OpenRouter; keep it out.
  delete process.env.OPENROUTER_API_KEY;
});

afterAll(() => {
  if (savedAnthropicKey === undefined) delete process.env.ANTHROPIC_API_KEY;
  else process.env.ANTHROPIC_API_KEY = savedAnthropicKey;
  if (savedOpenRouterKey === undefined) delete process.env.OPENROUTER_API_KEY;
  else process.env.OPENROUTER_API_KEY = savedOpenRouterKey;
});

beforeEach(() => {
  fetchMock.mockReset();
  globalLLMProviderHealth.reset();
});

describe('Anthropic catalog and pricing through the public API', () => {
  it('generateObject on Claude Sonnet 5.5 sends no forced tool and parses the JSON reply', async () => {
    route({ 'claude-sonnet-5-5': [textTurn('claude-sonnet-5-5', '{"answer":"ok"}')] });

    const result = await generateObject({
      model: 'anthropic:claude-sonnet-5-5',
      schema: z.object({ answer: z.string() }),
      prompt: 'Answer with ok.',
      fallbackProviders: [],
    });

    expect(result.object).toEqual({ answer: 'ok' });
    const [body] = postedBodies();
    // Sonnet 5.5 returns HTTP 400 on a forced tool_choice, so structured
    // output rides the prompt-only JSON path.
    expect(body.tool_choice).toBeUndefined();
    expect(body.tools).toBeUndefined();
  });

  it('reports Claude Opus 5.5 cache reads at 0.05x in result.usage.costUSD', async () => {
    route({
      'claude-opus-5-5': [
        textTurn(
          'claude-opus-5-5',
          'Hello.',
          { input_tokens: 1000, cache_read_input_tokens: 100000, cache_creation_input_tokens: 0 },
          500,
        ),
      ],
    });

    const result = await generateText({
      provider: 'anthropic',
      model: 'claude-opus-5-5',
      prompt: 'Say hello.',
      fallbackProviders: [],
    });

    // $0.004 input + $0.02 cache reads + $0.01 output.
    expect(result.usage.costUSD).toBeCloseTo(0.034, 6);
  });
});
