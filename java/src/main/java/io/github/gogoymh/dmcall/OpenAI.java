package io.github.gogoymh.dmcall;

import java.util.List;

/** OpenAI Chat Completions. Key: {@code OPENAI_API_KEY} (base URL: {@code OPENAI_BASE_URL}, optional). */
public final class OpenAI extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("openai", "https://api.openai.com/v1", List.of("OPENAI_BASE_URL"), List.of("OPENAI_API_KEY"), true);

    private OpenAI(Builder b) {
        super(b, KIND, "max_completion_tokens", null);
    }

    /** A provider for {@code model} with the key from the environment. */
    public OpenAI(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    /** Builds an {@link OpenAI}. */
    public static final class Builder extends ChatBuilder<Builder, OpenAI> {
        Builder(String model) {
            super(model);
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public OpenAI build() {
            return new OpenAI(this);
        }
    }
}
