package io.github.gogoymh.dmcall;

import java.util.List;

/**
 * Any Chat Completions endpoint with function tools: vLLM, Ollama, LM Studio, Together, Groq, ...
 *
 * <p>{@code OpenAICompatible.builder("llama3.1").baseUrl("http://localhost:11434/v1").build()}
 * needs no key.
 */
public final class OpenAICompatible extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("openai-compatible", null, List.of(), List.of(), false);

    private OpenAICompatible(Builder b) {
        super(b, KIND, "max_completion_tokens", null);
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    /** Builds an {@link OpenAICompatible}. */
    public static final class Builder extends ChatBuilder<Builder, OpenAICompatible> {
        Builder(String model) {
            super(model);
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public OpenAICompatible build() {
            return new OpenAICompatible(this);
        }
    }
}
