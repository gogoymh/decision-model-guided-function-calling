package io.github.gogoymh.dmcall;

import java.util.List;

/**
 * Meta Model API (Muse models such as {@code muse-spark-1.3}), OpenAI-compatible. Key:
 * {@code MODEL_API_KEY}.
 *
 * <p>Muse models are reasoning models: Meta rejects {@code reasoning_effort: "none"} for them, and a
 * {@code tool_choice} other than {@code "auto"}. This provider therefore sends neither
 * {@code reasoning_effort} nor {@code parallel_tool_calls} unless you set them.
 */
public final class Meta extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("meta", "https://api.meta.ai/v1", List.of(), List.of("MODEL_API_KEY"), true);

    private Meta(Builder b) {
        super(b, KIND, "max_completion_tokens", null);
    }

    /** A provider for {@code model} with the key from {@code MODEL_API_KEY}. */
    public Meta(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    /** Builds a {@link Meta}. Unlike the other Chat Completions builders, {@code parallel_tool_calls} is unset. */
    public static final class Builder extends ChatBuilder<Builder, Meta> {
        Builder(String model) {
            super(model);
            this.parallelToolCalls = null;
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public Meta build() {
            return new Meta(this);
        }
    }
}
