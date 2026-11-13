package io.github.gogoymh.dmcall;

import java.util.List;

/** xAI Grok, OpenAI-compatible. Key: {@code XAI_API_KEY}. */
public final class XAI extends ChatCompletionsProvider {
    static final Kind KIND = new Kind("xai", "https://api.x.ai/v1", List.of(), List.of("XAI_API_KEY"), true);

    private XAI(Builder b) {
        super(b, KIND, "max_tokens", null);
    }

    /** A provider for {@code model} with the key from the environment. */
    public XAI(String model) {
        this(builder(model));
    }

    public static Builder builder(String model) {
        return new Builder(model);
    }

    /** Builds an {@link XAI}. */
    public static final class Builder extends ChatBuilder<Builder, XAI> {
        Builder(String model) {
            super(model);
        }

        @Override
        protected Builder self() {
            return this;
        }

        @Override
        public XAI build() {
            return new XAI(this);
        }
    }
}
