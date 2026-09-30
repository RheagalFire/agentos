/**
 * @fileoverview GMI metaprompt tests: lifecycle state, one-at-a-time batches,
 * shutdown, turn_interval cadence, and the metaprompts of the shipped voice
 * assistant persona.
 *
 * Each case drives a real GMI, MetapromptExecutor, PromptEngine and
 * InMemoryWorkingMemory. Only the LLM provider is stubbed, along with the tool
 * orchestrator and the IUtilityAI dependency, which these turns use for
 * nothing beyond an empty tool list and JSON parsing. Metaprompt completions
 * are either answered at once or held open and settled by hand, so each case
 * controls when a background batch finishes relative to the turns around it.
 */
import { beforeAll, describe, expect, it, vi, type Mock } from 'vitest';

import { GMI } from '../GMI';
import {
  GMIBaseConfig,
  GMIInteractionType,
  GMIMood,
  GMIOutputChunk,
  GMIOutputChunkType,
  GMIPrimeState,
  GMITurnInput,
  ReasoningEntryType,
  ReasoningTraceEntry,
} from '../IGMI';
import type { IPersonaDefinition, MetaPromptDefinition } from '../personas/IPersonaDefinition';
import { getBuiltInPersona } from '../personas/definitions';
import { InMemoryWorkingMemory } from '../memory/InMemoryWorkingMemory';
import { PromptEngine } from '../../../core/llm/PromptEngine';
import type { AIModelProviderManager } from '../../../core/llm/providers/AIModelProviderManager';
import type {
  ChatMessage,
  IProvider,
  ModelCompletionOptions,
  ModelCompletionResponse,
} from '../../../core/llm/providers/IProvider';
import type { IUtilityAI } from '../../nlp/ai_utilities/IUtilityAI';
import type { IToolOrchestrator } from '../../../core/tools/IToolOrchestrator';

const PROVIDER_ID = 'mock-llm-provider';
const MODEL_ID = 'mock-model';
const REPLY_TEXT = 'Hello.';

type CompletionFn = (
  modelId: string,
  messages: ChatMessage[],
  options: ModelCompletionOptions,
) => Promise<ModelCompletionResponse>;

/** A metaprompt LLM call held open until the test settles it. */
interface PendingCompletion {
  resolve: (content: string) => void;
  reject: (error: Error) => void;
}

interface Harness {
  gmi: GMI;
  workingMemory: InMemoryWorkingMemory;
  /** The metaprompt LLM call (`IProvider.generateCompletion`). */
  generateCompletion: Mock<CompletionFn>;
  /** Metaprompt calls waiting for a result, in call order. Stays empty when auto-replying. */
  pending: PendingCompletion[];
}

/** A manual metaprompt, served by the executor's generic handler. */
const MANUAL_REFLECTION: MetaPromptDefinition = {
  id: 'reflect_now',
  promptTemplate: 'Reflect on {{recent_conversation}} and reply with JSON.',
  trigger: { type: 'manual' },
  modelId: MODEL_ID,
  providerId: PROVIDER_ID,
};

/** The self-reflection metaprompt, with a manual trigger so it never fires on its own. */
const TRAIT_ADJUSTMENT: MetaPromptDefinition = {
  id: 'gmi_self_trait_adjustment',
  promptTemplate: 'Evidence: {{evidence}}. Mood: {{current_mood}}. Reply with JSON.',
  trigger: { type: 'manual' },
  modelId: MODEL_ID,
  providerId: PROVIDER_ID,
};

let promptEngine: PromptEngine;

beforeAll(async () => {
  promptEngine = new PromptEngine();
  await promptEngine.initialize({
    defaultTemplateName: 'openai_chat',
    availableTemplates: {},
    tokenCounting: { strategy: 'estimated' },
    historyManagement: {
      defaultMaxMessages: 20,
      maxTokensForHistory: 4096,
      summarizationTriggerRatio: 0.8,
      preserveImportantMessages: true,
    },
    contextManagement: {
      maxRAGContextTokens: 2048,
      summarizationQualityTier: 'balanced',
      preserveSourceAttributionInSummary: true,
    },
    contextualElementSelection: {
      maxElementsPerType: {},
      defaultMaxElementsPerType: 3,
      priorityResolutionStrategy: 'highest_first',
      conflictResolutionStrategy: 'skip_conflicting',
    },
    // Caching would start an eviction interval that outlives the suite.
    performance: { enableCaching: false, cacheTimeoutSeconds: 60 },
  });
});

/** A non-streaming completion carrying `content`, as a metaprompt call returns it. */
function completion(content: string): ModelCompletionResponse {
  return {
    id: 'cmp-metaprompt',
    object: 'chat.completion',
    created: 0,
    modelId: MODEL_ID,
    choices: [{ index: 0, message: { role: 'assistant', content }, finishReason: 'stop' }],
    usage: { totalTokens: 1 },
  };
}

/** The reply every turn streams: one text delta, then the final chunk. */
async function* streamReply(): AsyncGenerator<ModelCompletionResponse, void, undefined> {
  yield {
    id: 'cmp-turn',
    object: 'chat.completion.chunk',
    created: 0,
    modelId: MODEL_ID,
    choices: [],
    responseTextDelta: REPLY_TEXT,
    isFinal: false,
  };
  yield {
    id: 'cmp-turn',
    object: 'chat.completion.chunk',
    created: 0,
    modelId: MODEL_ID,
    choices: [{ index: 0, message: { role: 'assistant', content: REPLY_TEXT }, finishReason: 'stop' }],
    usage: { totalTokens: 2, promptTokens: 1, completionTokens: 1 },
    isFinal: true,
  };
}

/**
 * Builds an initialized GMI for `persona` whose provider is registered as
 * `mock-llm-provider` and is also the GMI's default provider and model.
 *
 * @param persona - The persona to run.
 * @param autoReply - When set, every metaprompt call answers with this content
 *   at once. Otherwise each call waits in `pending` until the test settles it.
 */
async function createHarness(persona: IPersonaDefinition, autoReply?: string): Promise<Harness> {
  const pending: PendingCompletion[] = [];
  const generateCompletion = vi.fn<CompletionFn>(
    (): Promise<ModelCompletionResponse> =>
      autoReply !== undefined
        ? Promise.resolve(completion(autoReply))
        : new Promise<ModelCompletionResponse>((resolve, reject) => {
            pending.push({ resolve: (content) => resolve(completion(content)), reject });
          }),
  );
  const provider = {
    providerId: PROVIDER_ID,
    isInitialized: true,
    generateCompletion,
    generateCompletionStream: () => streamReply(),
  } as unknown as IProvider;

  const llmProviderManager = {
    getProvider: (providerId: string) => (providerId === PROVIDER_ID ? provider : undefined),
    getProviderForModel: () => provider,
    getDefaultProvider: () => provider,
    getModelInfo: async () => ({
      modelId: MODEL_ID,
      providerId: PROVIDER_ID,
      contextWindowSize: 128000,
      capabilities: ['chat'],
    }),
  } as unknown as AIModelProviderManager;

  const utilityAI = {
    parseJsonSafe: async (text: string) => {
      try {
        return JSON.parse(text);
      } catch {
        return null;
      }
    },
  } as unknown as IUtilityAI;

  const toolOrchestrator = {
    listAvailableTools: async () => [],
    processToolCall: vi.fn(),
  } as unknown as IToolOrchestrator;

  const workingMemory = new InMemoryWorkingMemory();
  const config: GMIBaseConfig = {
    workingMemory,
    promptEngine,
    llmProviderManager,
    utilityAI,
    toolOrchestrator,
    defaultLlmModelId: MODEL_ID,
    defaultLlmProviderId: PROVIDER_ID,
  };

  const gmi = new GMI();
  await gmi.initialize(persona, config);
  return { gmi, workingMemory, generateCompletion, pending };
}

function createPersona(metaPrompts: MetaPromptDefinition[]): IPersonaDefinition {
  return {
    id: 'metaprompt-lifecycle-persona',
    name: 'Metaprompt Lifecycle Persona',
    description: 'Persona for metaprompt lifecycle tests.',
    version: '1.0.0',
    baseSystemPrompt: 'You are a concise test assistant.',
    defaultModelId: MODEL_ID,
    defaultProviderId: PROVIDER_ID,
    metaPrompts,
  };
}

function userTurn(interactionId: string, overrides: Partial<GMITurnInput> = {}): GMITurnInput {
  return {
    interactionId,
    userId: 'user-1',
    type: GMIInteractionType.TEXT,
    content: `Message for ${interactionId}`,
    ...overrides,
  };
}

/** Consumes the rest of a turn's stream and returns the chunks it yielded. */
async function drainTurn(stream: AsyncIterable<GMIOutputChunk>): Promise<GMIOutputChunk[]> {
  const chunks: GMIOutputChunk[] = [];
  for await (const chunk of stream) chunks.push(chunk);
  return chunks;
}

async function runTurn(
  gmi: GMI,
  interactionId: string,
  overrides: Partial<GMITurnInput> = {},
): Promise<GMIOutputChunk[]> {
  return drainTurn(gmi.processTurnStream(userTurn(interactionId, overrides)));
}

function textOf(chunks: GMIOutputChunk[]): string {
  return chunks
    .filter((chunk) => chunk.type === GMIOutputChunkType.TEXT_DELTA)
    .map((chunk) => String(chunk.content))
    .join('');
}

function traceOf(gmi: GMI, type: ReasoningEntryType): ReasoningTraceEntry[] {
  return gmi.getReasoningTrace().entries.filter((entry) => entry.type === type);
}

/** Lets queued promise callbacks and zero-delay timers run. */
function settle(): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, 0));
}

async function waitForTrace(gmi: GMI, type: ReasoningEntryType, count: number): Promise<void> {
  await vi.waitFor(() => expect(traceOf(gmi, type)).toHaveLength(count));
  // The queue releases its slot a few callbacks after the trace entry lands.
  await settle();
}

async function currentMood(harness: Harness): Promise<unknown> {
  return (await harness.gmi.getWorkingMemorySnapshot()).currentGmiMood;
}

/** Sets the working-memory flag that fires a manual metaprompt on the next turn. */
async function arm(harness: Harness, metapromptId: string = MANUAL_REFLECTION.id): Promise<void> {
  await harness.workingMemory.set(`manual_trigger_${metapromptId}`, true);
}

/** A `turn_interval` metaprompt, served by the executor's generic handler. */
function intervalMetaprompt(intervalTurns: number): MetaPromptDefinition {
  return {
    id: 'cadence_probe',
    promptTemplate: 'Review {{recent_conversation}} and reply with JSON.',
    trigger: { type: 'turn_interval', intervalTurns },
    modelId: MODEL_ID,
    providerId: PROVIDER_ID,
  };
}

/** Ids of the turns whose metaprompt check fired, in order. */
function triggeredTurnIds(gmi: GMI): string[] {
  return traceOf(gmi, ReasoningEntryType.SELF_REFLECTION_TRIGGERED).map((entry) =>
    String(entry.details?.turnId),
  );
}

function warningsMentioning(gmi: GMI, text: string): ReasoningTraceEntry[] {
  return traceOf(gmi, ReasoningEntryType.WARNING).filter((entry) => entry.message.includes(text));
}

describe('GMI metaprompt lifecycle', () => {
  it('a finished metaprompt batch leaves the GMI ready for the next turn', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION]));
    await arm(h);

    await runTurn(h.gmi, 'turn-1');
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));

    h.pending[0].resolve(JSON.stringify({ updatedGmiMood: GMIMood.FOCUSED }));
    await waitForTrace(h.gmi, ReasoningEntryType.SELF_REFLECTION_COMPLETE, 1);

    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
    expect(await currentMood(h)).toBe(GMIMood.FOCUSED);
    expect(textOf(await runTurn(h.gmi, 'turn-2'))).toBe(REPLY_TEXT);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
  });

  it('a metaprompt that settles during a turn leaves that turn running', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION, TRAIT_ADJUSTMENT]));

    // turn-1's background batch settles while turn-2 is streaming.
    await arm(h);
    await runTurn(h.gmi, 'turn-1');
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));
    const turn2 = h.gmi.processTurnStream(userTurn('turn-2'));
    expect((await turn2.next()).done).toBe(false);
    h.pending[0].resolve('not json');
    await waitForTrace(h.gmi, ReasoningEntryType.SELF_REFLECTION_COMPLETE, 1);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.PROCESSING);
    await drainTurn(turn2);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);

    // A manual self-reflection is in flight when turn-3 starts, and settles mid-turn.
    const reflection = h.gmi._triggerAndProcessSelfReflection();
    await vi.waitFor(() => expect(h.pending).toHaveLength(2));
    const turn3 = h.gmi.processTurnStream(userTurn('turn-3'));
    expect((await turn3.next()).done).toBe(false);
    h.pending[1].resolve('{}');
    await reflection;
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.PROCESSING);
    await drainTurn(turn3);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
  });

  it('runs one batch at a time and keeps going after a failed batch', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION]));

    await arm(h);
    await runTurn(h.gmi, 'turn-1');
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));
    await arm(h);
    await runTurn(h.gmi, 'turn-2');
    await settle();
    // turn-2's batch waits behind turn-1's, whose call is still open.
    expect(h.generateCompletion).toHaveBeenCalledTimes(1);

    h.pending[0].reject(new Error('provider unavailable'));
    await vi.waitFor(() => expect(h.pending).toHaveLength(2));
    h.pending[1].resolve(JSON.stringify({ updatedGmiMood: GMIMood.CURIOUS }));
    await waitForTrace(h.gmi, ReasoningEntryType.SELF_REFLECTION_COMPLETE, 2);

    const batchEvents = h.gmi
      .getReasoningTrace()
      .entries.filter(
        (entry) =>
          entry.type === ReasoningEntryType.SELF_REFLECTION_START ||
          entry.type === ReasoningEntryType.SELF_REFLECTION_COMPLETE,
      )
      .map((entry) => entry.type);
    expect(batchEvents).toEqual([
      ReasoningEntryType.SELF_REFLECTION_START,
      ReasoningEntryType.SELF_REFLECTION_COMPLETE,
      ReasoningEntryType.SELF_REFLECTION_START,
      ReasoningEntryType.SELF_REFLECTION_COMPLETE,
    ]);
    const failures = traceOf(h.gmi, ReasoningEntryType.ERROR).filter((entry) =>
      entry.message.includes("Metaprompt 'reflect_now' failed"),
    );
    expect(failures).toHaveLength(1);
    expect(await currentMood(h)).toBe(GMIMood.CURIOUS);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
  });

  it('skips a manual self-reflection while metaprompt work runs, and never changes lifecycle state', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION, TRAIT_ADJUSTMENT]));
    await arm(h);
    await runTurn(h.gmi, 'turn-1');
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));

    const skipped = h.gmi._triggerAndProcessSelfReflection();
    await settle();
    expect(traceOf(h.gmi, ReasoningEntryType.SELF_REFLECTION_SKIPPED).map((entry) => entry.message)).toEqual([
      'Self-reflection already in progress.',
    ]);
    expect(h.generateCompletion).toHaveBeenCalledTimes(1);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
    await skipped;

    h.pending[0].resolve('not json');
    await waitForTrace(h.gmi, ReasoningEntryType.SELF_REFLECTION_COMPLETE, 1);

    // With the queue idle, the same request runs.
    const reflection = h.gmi._triggerAndProcessSelfReflection();
    await vi.waitFor(() => expect(h.pending).toHaveLength(2));
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
    h.pending[1].resolve(JSON.stringify({ updatedGmiMood: GMIMood.ANALYTICAL }));
    await reflection;
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.READY);
    expect(await currentMood(h)).toBe(GMIMood.ANALYTICAL);
  });
});

describe('GMI shutdown with metaprompt work in flight', () => {
  it('waits for a running batch to store its updates before closing working memory', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION]));
    const setSpy = vi.spyOn(h.workingMemory, 'set');
    const closeSpy = vi.spyOn(h.workingMemory, 'close');
    await arm(h);
    await runTurn(h.gmi, 'turn-1');
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));

    const shuttingDown = h.gmi.shutdown();
    await settle();
    expect(closeSpy).not.toHaveBeenCalled();

    h.pending[0].resolve(JSON.stringify({ updatedGmiMood: GMIMood.FOCUSED }));
    await shuttingDown;

    const moodWrite = setSpy.mock.calls.findIndex(
      ([key, value]) => key === 'currentGmiMood' && value === GMIMood.FOCUSED,
    );
    expect(moodWrite).toBeGreaterThanOrEqual(0);
    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect(setSpy.mock.invocationCallOrder[moodWrite]).toBeLessThan(closeSpy.mock.invocationCallOrder[0]);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.SHUTDOWN);
  });

  it('stops waiting for a metaprompt call that never settles', async () => {
    const h = await createHarness(createPersona([MANUAL_REFLECTION]));
    const closeSpy = vi.spyOn(h.workingMemory, 'close');
    await arm(h);
    await runTurn(h.gmi, 'turn-1');
    await vi.waitFor(() => expect(h.pending).toHaveLength(1));

    vi.useFakeTimers({ toFake: ['setTimeout', 'clearTimeout'] });
    try {
      const shuttingDown = h.gmi.shutdown();
      await vi.advanceTimersByTimeAsync(5000);
      await shuttingDown;
    } finally {
      vi.useRealTimers();
    }

    expect(closeSpy).toHaveBeenCalledTimes(1);
    expect(h.gmi.getCurrentState()).toBe(GMIPrimeState.SHUTDOWN);
  });
});

describe('turn_interval cadence', () => {
  it.each([
    { intervalTurns: 1, turns: 3, fired: ['turn-1', 'turn-2', 'turn-3'] },
    { intervalTurns: 3, turns: 6, fired: ['turn-3', 'turn-6'] },
  ])(
    'intervalTurns $intervalTurns fires on every Nth user turn',
    async ({ intervalTurns, turns, fired }) => {
      const h = await createHarness(createPersona([intervalMetaprompt(intervalTurns)]), 'not json');
      for (let turn = 1; turn <= turns; turn += 1) {
        await runTurn(h.gmi, `turn-${turn}`);
      }
      expect(triggeredTurnIds(h.gmi)).toEqual(fired);
    },
  );

  const nonUserTurns: Array<[string, (gmi: GMI) => Promise<unknown>]> = [
    [
      'a tool continuation',
      (gmi) =>
        gmi.handleToolResults(
          [{ toolCallId: 'external-1', toolName: 'lookup', output: { ok: true } }],
          'user-1',
        ),
    ],
    [
      'a system message turn',
      (gmi) =>
        runTurn(gmi, 'system-1', { type: GMIInteractionType.SYSTEM_MESSAGE, content: 'Session resumed.' }),
    ],
    [
      'a tool response turn',
      (gmi) =>
        runTurn(gmi, 'tool-response-1', {
          type: GMIInteractionType.TOOL_RESPONSE,
          content: [{ toolCallId: 'external-2', toolName: 'lookup', output: { ok: true } }],
        }),
    ],
  ];

  it.each(nonUserTurns)(
    '%s neither counts toward nor fires a turn_interval metaprompt',
    async (_label, runNonUserTurn) => {
      const h = await createHarness(createPersona([intervalMetaprompt(2)]), 'not json');

      await runTurn(h.gmi, 'turn-1');
      await runNonUserTurn(h.gmi);
      for (const turnId of ['turn-2', 'turn-3', 'turn-4']) {
        await runTurn(h.gmi, turnId);
      }

      expect(triggeredTurnIds(h.gmi)).toEqual(['turn-2', 'turn-4']);
    },
  );

  it.each([[0], [Number.NaN], [undefined]])(
    'intervalTurns %s never fires and is reported once',
    async (intervalTurns) => {
      const h = await createHarness(
        createPersona([intervalMetaprompt(intervalTurns as number)]),
        'not json',
      );

      await runTurn(h.gmi, 'turn-1');
      await runTurn(h.gmi, 'turn-2');

      expect(triggeredTurnIds(h.gmi)).toEqual([]);
      expect(h.generateCompletion).not.toHaveBeenCalled();
      expect(warningsMentioning(h.gmi, 'intervalTurns')).toHaveLength(1);
    },
  );
});

describe('metaprompts the executor cannot run', () => {
  it('reports an unsupported trigger type once and never fires it', async () => {
    // Persona JSON is cast, not type-checked, so this shape reaches the executor.
    const preResponse = {
      id: 'voice_polish',
      promptTemplate: 'Polish {{recent_conversation}} and reply with JSON.',
      trigger: { type: 'pre_response' },
      modelId: MODEL_ID,
      providerId: PROVIDER_ID,
    } as unknown as MetaPromptDefinition;
    const h = await createHarness(createPersona([preResponse]), 'not json');

    await runTurn(h.gmi, 'turn-1');
    await runTurn(h.gmi, 'turn-2');

    expect(triggeredTurnIds(h.gmi)).toEqual([]);
    expect(h.generateCompletion).not.toHaveBeenCalled();
    expect(warningsMentioning(h.gmi, "'voice_polish'").map((entry) => entry.message)).toEqual([
      "Metaprompt 'voice_polish' will not run: trigger type 'pre_response' is not supported (expected turn_interval, event_based or manual).",
    ]);
  });
});

describe('shipped voice assistant persona', () => {
  it('runs its trait adjustment every 7 user turns on the GMI default model and applies the result', async () => {
    const persona = getBuiltInPersona('voice_assistant_persona');
    if (!persona) throw new Error('voice_assistant_persona is not a built-in persona');
    const h = await createHarness(
      persona,
      JSON.stringify({ updatedGmiMood: GMIMood.FOCUSED, adjustmentRationale: 'The user wants short spoken answers.' }),
    );

    for (let turn = 1; turn <= 7; turn += 1) {
      await runTurn(h.gmi, `turn-${turn}`);
    }
    expect(triggeredTurnIds(h.gmi)).toEqual(['turn-7']);
    await waitForTrace(h.gmi, ReasoningEntryType.SELF_REFLECTION_COMPLETE, 1);

    // The persona pins no model, so the reflection runs on the GMI's configured default.
    expect(h.generateCompletion).toHaveBeenCalledTimes(1);
    const [modelId, messages, options] = h.generateCompletion.mock.calls[0];
    expect(modelId).toBe(MODEL_ID);
    expect(options).toMatchObject({ maxTokens: 400, responseFormat: { type: 'json_object' } });
    const prompt = String(messages[0]?.content);
    expect(prompt).toContain('Current mood: helpful_engaged.');
    expect(prompt).not.toMatch(/\{\{\s*\w+\s*\}\}/);

    expect(await currentMood(h)).toBe(GMIMood.FOCUSED);
    expect(warningsMentioning(h.gmi, 'will not run')).toEqual([]);
  });
});
