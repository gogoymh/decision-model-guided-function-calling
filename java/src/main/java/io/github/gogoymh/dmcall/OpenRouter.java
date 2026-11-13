package io.github.gogoymh.dmcall;

import java.util.List;

/** OpenRouter, one OpenAI-compatible endpoint for many models. Key: {@code OPENROUTER_API_KEY}. */
public final class OpenRouter extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("openrouter", "https://openrouter.ai/api/v1", List.of(), List.of("OPENROUTER_API_KEY"), true);

    private OpenRouter(Builder b) {
        super(b, KIND, "max_tokens", null);
    }

    /** A provider for {@code model} with the key from the environment. */
    public OpenRouter(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    /** Builds an {@link OpenRouter}. */
    public static final class Builder extends ChatBuilder<Builder, OpenRouter> {
        Builder(String model) {
            super(model);
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public OpenRouter build() {
            return new OpenRouter(this);
        }
    }
}
